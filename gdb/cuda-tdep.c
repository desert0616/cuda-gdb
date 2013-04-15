/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2012 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include <time.h>
#include <sys/stat.h>
#include <ctype.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <signal.h>
#include <execinfo.h>

#include "defs.h"
#include "arch-utils.h"
#include "command.h"
#include "dummy-frame.h"
#include "dwarf2-frame.h"
#include "doublest.h"
#include "floatformat.h"
#include "frame.h"
#include "frame-base.h"
#include "frame-unwind.h"
#include "inferior.h"
#include "gdbcmd.h"
#include "gdbcore.h"
#include "objfiles.h"
#include "osabi.h"
#include "regcache.h"
#include "regset.h"
#include "symfile.h"
#include "symtab.h"
#include "target.h"
#include "value.h"
#include "dis-asm.h"
#include "source.h"
#include "block.h"
#include "gdb/signals.h"
#include "gdbthread.h"
#include "language.h"
#include "demangle.h"
#include "main.h"
#include "target.h"
#include "valprint.h"
#include "user-regs.h"
#include "linux-tdep.h"
#include "exec.h"

#include "gdb_assert.h"
#include "gdb_string.h"

#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-iterator.h"
#include "cuda-modules.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "cuda-textures.h"
#include "libbfd.h"
#include "mach-o.h"
#include "cuda-regmap.h"

/*----------------------------------------- Globals ---------------------------------------*/

struct gdbarch *cuda_gdbarch = NULL;
CuDim3 gridDim  = { 0, 0, 0};
CuDim3 blockDim = { 0, 0, 0};
bool cuda_initialized = false;
static bool inferior_in_debug_mode = false;
static char cuda_gdb_session_dir[CUDA_GDB_TMP_BUF_SIZE] = {0};
static uint32_t cuda_gdb_session_id = 0;

/* For Mac OS X */
bool cuda_platform_supports_tid (void)
{
#if defined(linux) && defined(SYS_gettid)
    return true;
#else
    return false;
#endif
}

int
cuda_gdb_get_tid (ptid_t ptid)
{
  if (cuda_platform_supports_tid ())
    return TIDGET (ptid);
  else
    return PIDGET (ptid);
}


/* CUDA - skip prologue */
bool cuda_producer_is_open64;

/* CUDA architecture specific information.  */
struct gdbarch_tdep
{
  int num_regs;
  int num_pseudo_regs;

  /* Pointer size: 32-bits on i686, 64-bits on x86_64. So that we do not need
     to have 2 versions of the cuda-tdep.c file */
  int ptr_size;

  /* Registers */

  /* always */
  int pc_regnum;
  /* ABI only */
  int sp_regnum;
  int first_rv_regnum;
  int last_rv_regnum;
  int rz_regnum;
  int max_reg_rv_size;

  /* Pseudo-Registers */

  /* Special register to indicate to look at the regmap search result */
  int special_regnum;

  /* Register number to tell the debugger that no valid register was found.
     Used to avoid returning errors and having nice warning messages and
     consistent garbage return values instead (zero sounds good).
     invalid_lo_regnum is used for variables that aren't live and are
     stored in a single register.  The combination of invalid_lo_regnum
     and invalid_hi_regnum is needed when a variable isn't live and is
     stored in multiple registers. */
  int invalid_lo_regnum;
  int invalid_hi_regnum;
};


int
cuda_special_regnum (struct gdbarch *gdbarch)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return tdep->special_regnum;
}

int
cuda_pc_regnum (struct gdbarch *gdbarch)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return tdep->pc_regnum;
}

/* CUDA - siginfo */
static int cuda_signo = 0;

static void
cuda_siginfo_trace (char *fmt, ...)
{
  va_list ap;

  if (cuda_options_debug_siginfo ())
    {
      va_start (ap, fmt);
      fprintf (stderr, "[CUDAGDB] siginfo -- ");
      vfprintf (stderr, fmt, ap);
      fprintf (stderr, "\n");
      fflush (stderr);
    }
}
void
cuda_set_signo (int signo)
{
  cuda_signo = signo;

  cuda_siginfo_trace ("CUDA siginfo set to %d", signo);
}

int
cuda_get_signo (void)
{
  return cuda_signo;
}


/*---------------------------------------- Routines ---------------------------------------*/

int
cuda_inferior_word_size ()
{
#ifdef __linux__
  gdb_assert (0); // not implemented
#else
  unsigned long cputype = bfd_mach_o_cputype (exec_bfd);
  switch (cputype)
    {
      case BFD_MACH_O_CPU_TYPE_I386:
        return 4;
      case BFD_MACH_O_CPU_TYPE_X86_64:
        return 8;
      default:
        error (_("Unsupported CPU type: 0x%lx\n"), cputype);
    }
#endif

  gdb_assert (0);
  return -1;
}

char *
cuda_find_kernel_name_from_pc (CORE_ADDR pc, bool demangle)
{
  char *demangled = NULL, *name = NULL;
  static char *unknown = "??";
  struct symbol *kernel = NULL;
  enum language lang = language_unknown;
  struct minimal_symbol *msymbol = lookup_minimal_symbol_by_pc (pc);

  /* find the mangled name */
  kernel = find_pc_function (pc);
  if (kernel && msymbol != NULL &&
      SYMBOL_VALUE_ADDRESS (msymbol) > BLOCK_START (SYMBOL_BLOCK_VALUE (kernel)))
    {
      name = SYMBOL_LINKAGE_NAME (msymbol);
      lang = SYMBOL_LANGUAGE (msymbol);
    }
  else if (kernel)
    {
      name = SYMBOL_LINKAGE_NAME (kernel);
      lang = SYMBOL_LANGUAGE (kernel);
    }
  else if (msymbol != NULL)
    {
      name = SYMBOL_LINKAGE_NAME (msymbol);
      lang = SYMBOL_LANGUAGE (msymbol);
    }

  /* process the mangled name */
  if (!name)
    return unknown;
  else if (demangle)
    {
      demangled = language_demangle (language_def (lang), name, DMGL_ANSI);
      if (demangled)
        return demangled;
      else
        return name;
    }
  else
    return name;
}

void
cuda_trace (char *fmt, ...)
{
  va_list ap;

  if (cuda_options_debug_general ())
    {
      va_start (ap, fmt);
      fprintf (stderr, "[CUDAGDB] ");
      vfprintf (stderr, fmt, ap);
      fprintf (stderr, "\n");
      fflush (stderr);
    }
}

bool
cuda_breakpoint_hit_p (cuda_clock_t clock)
{
  cuda_iterator itr;
  cuda_coords_t c, filter = CUDA_WILDCARD_COORDS;
  bool breakpoint_hit = false;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, &filter,
                               CUDA_SELECT_VALID | CUDA_SELECT_BKPT);

  for (cuda_iterator_start (itr);
       !cuda_iterator_end (itr);
       cuda_iterator_next (itr))
    {
      /* if we hit a breakpoint at an earlier time, we do not report it again. */
      c = cuda_iterator_get_current (itr);
      if (lane_get_timestamp (c.dev, c.sm, c.wp, c.ln) < clock)
        continue;

      breakpoint_hit = true;
      break;
    }

  cuda_iterator_destroy (itr);

  return breakpoint_hit;
}

bool
cuda_exception_hit_p (cuda_exception_t *exception)
{
  CUDBGException_t exception_type = CUDBG_EXCEPTION_NONE;
  cuda_coords_t c, filter = CUDA_WILDCARD_COORDS;
  cuda_iterator itr;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, &filter,
                               CUDA_SELECT_VALID | CUDA_SELECT_EXCPT);

  for (cuda_iterator_start (itr);
       !cuda_iterator_end (itr);
       cuda_iterator_next (itr))
    {
      c = cuda_iterator_get_current (itr);
      exception_type = lane_get_exception (c.dev, c.sm, c.wp, c.ln);
      break;
    }

  cuda_iterator_destroy (itr);

  switch (exception_type)
    {
    case CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS:
      exception->value = TARGET_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW:
      exception->value = TARGET_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW:
      exception->value = TARGET_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
      exception->value = TARGET_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
      exception->value = TARGET_SIGNAL_CUDA_WARP_OUT_OF_RANGE_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
      exception->value = TARGET_SIGNAL_CUDA_WARP_MISALIGNED_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
      exception->value = TARGET_SIGNAL_CUDA_WARP_INVALID_ADDRESS_SPACE;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_INVALID_PC:
      exception->value = TARGET_SIGNAL_CUDA_WARP_INVALID_PC;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
      exception->value = TARGET_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
      exception->value = TARGET_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ASSERT:
      exception->value = TARGET_SIGNAL_CUDA_WARP_ASSERT;
      exception->valid = true;
      exception->recoverable = true;
      break;
    case CUDBG_EXCEPTION_NONE:
      break;
    case CUDBG_EXCEPTION_UNKNOWN:
    default:
      /* If for some reason the device encounters an unknown exception, we
         still need to halt the chip and allow state inspection.  Just emit
         a warning indicating this is something that was unexpected, but
         handle it as a normal device exception. */
      warning ("Encountered unhandled device exception (%d)\n", exception_type);
      exception->value = TARGET_SIGNAL_CUDA_UNKNOWN_EXCEPTION;
      exception->valid = true;
      exception->recoverable = false;
      break;
    }

  return exception_type != CUDBG_EXCEPTION_NONE;
}

const char *
cuda_exception_type_to_name (CUDBGException_t exception_type)
{
  switch (exception_type)
    {
    case CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS);
    case CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION);
    case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_OUT_OF_RANGE_ADDRESS);
    case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_MISALIGNED_ADDRESS);
    case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_INVALID_ADDRESS_SPACE);
    case CUDBG_EXCEPTION_WARP_INVALID_PC:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_INVALID_PC);
    case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS);
    case CUDBG_EXCEPTION_WARP_ASSERT:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_WARP_ASSERT);
    default:
      return target_signal_to_string (TARGET_SIGNAL_CUDA_UNKNOWN_EXCEPTION);
    }
}

void
cuda_update_convenience_variables (void)
{
  uint32_t num_dev, num_sm, num_wp, num_ln, num_reg, num_present_kernels, call_depth;
  uint32_t syscall_call_depth;
  uint64_t kernel_id;
  CuDim3 grid_dim;
  CuDim3 block_dim;
  cuda_coords_t current;
  kernels_t kernels;
  kernel_t kernel;
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type_uint32 = builtin_type (gdbarch)->builtin_uint32;
  struct type *type_uint64 = builtin_type (gdbarch)->builtin_uint32;
  struct value *mark = NULL;

  if (cuda_focus_is_device ())
    {
      cuda_coords_get_current (&current);
      kernels   = device_get_kernels (current.dev);
      kernel    = warp_get_kernel (current.dev, current.sm, current.wp);

      num_dev   = cuda_system_get_num_devices ();
      num_sm    = device_get_num_sms (current.dev);
      num_wp    = device_get_num_warps (current.dev);
      num_ln    = device_get_num_lanes (current.dev);
      num_reg   = device_get_num_registers (current.dev);
      grid_dim  = kernel_get_grid_dim (kernel);
      block_dim = kernel_get_block_dim (kernel);
      kernel_id = kernel_get_id (kernel);
      num_present_kernels = kernels_get_num_present_kernels (kernels);
      call_depth = lane_get_call_depth (current.dev, current.sm, current.wp, current.ln);
      syscall_call_depth = lane_get_syscall_call_depth(current.dev, current.sm, current.wp, current.ln);
    }
  else
    {
      num_dev   = CUDA_INVALID;
      num_sm    = CUDA_INVALID;
      num_wp    = CUDA_INVALID;
      num_ln    = CUDA_INVALID;
      num_reg   = CUDA_INVALID;
      grid_dim  = (CuDim3){ CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
      block_dim = (CuDim3){ CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
      kernel_id = CUDA_INVALID;
      current   = CUDA_INVALID_COORDS;
      num_present_kernels = CUDA_INVALID;
      call_depth = CUDA_INVALID;
      syscall_call_depth = CUDA_INVALID;
    }

  mark = value_mark ();

  set_internalvar (lookup_internalvar ("cuda_num_devices"),
                   value_from_longest (type_uint32, (LONGEST) num_dev));
  set_internalvar (lookup_internalvar ("cuda_num_sm"),
                   value_from_longest (type_uint32, (LONGEST) num_sm));
  set_internalvar (lookup_internalvar ("cuda_num_warps"),
                   value_from_longest (type_uint32, (LONGEST) num_wp));
  set_internalvar (lookup_internalvar ("cuda_num_lanes"),
                   value_from_longest (type_uint32, (LONGEST) num_ln));
  set_internalvar (lookup_internalvar ("cuda_num_registers"),
                   value_from_longest (type_uint32, (LONGEST) num_reg));
  set_internalvar (lookup_internalvar ("cuda_grid_dim_x"),
                   value_from_longest (type_uint32, (LONGEST) grid_dim.x));
  set_internalvar (lookup_internalvar ("cuda_grid_dim_y"),
                   value_from_longest (type_uint32, (LONGEST) grid_dim.y));
  set_internalvar (lookup_internalvar ("cuda_grid_dim_z"),
                   value_from_longest (type_uint32, (LONGEST) grid_dim.z));
  set_internalvar (lookup_internalvar ("cuda_block_dim_x"),
                   value_from_longest (type_uint32, (LONGEST) block_dim.x));
  set_internalvar (lookup_internalvar ("cuda_block_dim_y"),
                   value_from_longest (type_uint32, (LONGEST) block_dim.y));
  set_internalvar (lookup_internalvar ("cuda_block_dim_z"),
                   value_from_longest (type_uint32, (LONGEST) block_dim.z));
  set_internalvar (lookup_internalvar ("cuda_focus_device"),
                   value_from_longest (type_uint32, (LONGEST) current.dev));
  set_internalvar (lookup_internalvar ("cuda_focus_sm"),
                   value_from_longest (type_uint32, (LONGEST) current.sm));
  set_internalvar (lookup_internalvar ("cuda_focus_warp"),
                   value_from_longest (type_uint32, (LONGEST) current.wp));
  set_internalvar (lookup_internalvar ("cuda_focus_lane"),
                   value_from_longest (type_uint32, (LONGEST) current.ln));
  set_internalvar (lookup_internalvar ("cuda_focus_grid"),
                   value_from_longest (type_uint32, (LONGEST) current.gridId));
  set_internalvar (lookup_internalvar ("cuda_focus_kernel_id"),
                   value_from_longest (type_uint64, (LONGEST) current.kernelId));
  set_internalvar (lookup_internalvar ("cuda_focus_block_x"),
                   value_from_longest (type_uint32, (LONGEST) current.blockIdx.x));
  set_internalvar (lookup_internalvar ("cuda_focus_block_y"),
                   value_from_longest (type_uint32, (LONGEST) current.blockIdx.y));
  set_internalvar (lookup_internalvar ("cuda_focus_thread_z"),
                   value_from_longest (type_uint32, (LONGEST) current.blockIdx.z));
  set_internalvar (lookup_internalvar ("cuda_focus_thread_x"),
                   value_from_longest (type_uint32, (LONGEST) current.threadIdx.x));
  set_internalvar (lookup_internalvar ("cuda_focus_thread_y"),
                   value_from_longest (type_uint32, (LONGEST) current.threadIdx.y));
  set_internalvar (lookup_internalvar ("cuda_focus_thread_z"),
                   value_from_longest (type_uint32, (LONGEST) current.threadIdx.z));
  set_internalvar (lookup_internalvar ("cuda_num_present_kernels"),
                   value_from_longest (type_uint32, (LONGEST) num_present_kernels));
  set_internalvar (lookup_internalvar ("cuda_call_depth"),
                   value_from_longest (type_uint32, (LONGEST) call_depth));
  set_internalvar (lookup_internalvar ("cuda_syscall_call_depth"),
                   value_from_longest (type_uint32, (LONGEST) syscall_call_depth));

  /* Free the temporary values */
  value_free_to_mark (mark);
}

/* Return the name of register REGNUM. */
static const char*
cuda_register_name (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  uint32_t device_num_regs, sp_regnum, offset;
  static const int size = CUDA_GDB_TMP_BUF_SIZE;
  static char buf[CUDA_GDB_TMP_BUF_SIZE];
  int d, i;
  bool high;
  regmap_t regmap;

  /* Ignore registers not supported by this device */
  device_num_regs = device_get_num_registers (cuda_current_device ());

  /* Single SASS register */
  if (regnum < device_num_regs)
    {
      snprintf (buf, sizeof (buf), "R%d", regnum);
      return buf;
    }

  /* The PC register */
  if (regnum == tdep->pc_regnum)
    {
      return "pc";
    }

  /* Invalid register */
  if (regnum == tdep->invalid_lo_regnum ||
      regnum == tdep->invalid_hi_regnum)
    {
      snprintf (buf, sizeof (buf), "(dummy internal register)");
      return buf;
    }

  /* Not a special register (everything else) */
  if (regnum != tdep->special_regnum)
    return NULL;

  /* The special CUDA register: stored in the regmap. */
  regmap = regmap_get_search_result ();
  d = 0;
  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      if (i > 0)
        d += snprintf (buf + d, size - 1 - d, "/$");

      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "R%d", regnum);
            break;

          case REG_CLASS_MEM_LOCAL:
            offset = regmap_get_offset (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "(spilled @ 0x%x)", offset);
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "(spilled @ [R%d]+0x%x)",
                           sp_regnum,  offset);
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            d += snprintf (buf + d, size - 1 - d, "R%d.%s", regnum, high ? "hi" : "lo");
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }

  return buf;
}


static struct type *
cuda_register_type (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);

  if (regnum == tdep->special_regnum || regnum == tdep->pc_regnum)
    return builtin_type (gdbarch)->builtin_int64;
  else
    return builtin_type (gdbarch)->builtin_int32;
}

/*
 * Copy the DWARF register string that represents a CUDA register
 */
bool
cuda_get_dwarf_register_string (reg_t reg, char *deviceReg, size_t sz)
{
  static const int size = sizeof (ULONGEST);
  char buffer[size+1];
  char *p = NULL;
  bool isDeviceReg = false;
  int i;

  /* Is reg a virtual device register or a host register? Let's look
     at the encoding to determine it. Example:

     as char[4]      as uint32_t
     device reg %r4 :      "\04r%"     0x00257234
     host reg  4    : "\0\0\0\004"     0x00000004

     Therefore, as long as the host uses less than 256 registers, we
     can safely assume that if the uint32_t value of reg is larger
     than 0xff and the first character is a '%', we are dealing with
     a virtual device register (a virtual device register is made of
     at least 2 characters). */
  if (reg > 0xff)
    {
      /* if this is a device register, the register name string,
         originally encoded as ULEB128, has been decoded as an
         unsigned integer. The order of the characters has to be
         reversed in order to be read as a standard string. */
      memset (buffer, 0, size+1);
      for (i = 0; i < size; ++i)
        {
          buffer[size-1-i] = reg & 0xff;
          reg = reg >> 8;
        }

      /* find the first character of the string */
      p = buffer;
      while (*p == 0)
        ++p;

      /* copy the result if we are dealing with a device register. */
      if (p[0] == '%')
        {
          isDeviceReg = true;
          if (deviceReg)
            strncpy (deviceReg, p, sz);
        }
    }

  return !isDeviceReg;
}

/* Turns a virtual device PC to a physical one (currently only used
   when mapping a virtual PTX register to a physical device register).
   This is because the register map requires a physical PC that is
   an offset from the kernel entry function.

   NOTE:  For the ABI, this assumes that subroutine cloning is in place,
   and that each subroutine is offset from a kernel entry point.  This
   will need adjustment (which will be an improvement) if/when the
   compiler changes this. */
static CORE_ADDR
cuda_pc_virt_to_phys (CORE_ADDR pc)
{
  CORE_ADDR device_entry_frame_pc;  /* PC of outermost device frame (kernel) */
  CORE_ADDR device_entry_frame_base; /* Base address of device_entry_frame */
  cuda_coords_t c;

  if (cuda_current_active_elf_image_uses_abi ())
    /* ABI Compilation - need to unwind to the entry frame */
    {
      int call_depth = 0;

      /* Read the call depth on the device */
      cuda_coords_get_current (&c);

      call_depth = lane_get_call_depth (c.dev, c.sm, c.wp, c.ln);

      if (!call_depth)
        /* If there isn't a call depth, just set the entry pc to the input */
        device_entry_frame_pc = pc;
      else
        /* Find the entry frame's PC from the stack */
        device_entry_frame_pc =
            (CORE_ADDR) lane_get_virtual_return_address
              (c.dev, c.sm, c.wp, c.ln, call_depth - 1);
    }
  else
    /* No ABI - there is only one frame */
    device_entry_frame_pc = pc;

  /* Get the base VA for the kernel. */
  device_entry_frame_base = get_pc_function_start (device_entry_frame_pc);

  gdb_assert (pc >= device_entry_frame_base);
  return pc - device_entry_frame_base;
}

/* The following globals/functions are used to work around
   an issue with variable locations when they are stored
   in a device register.  For ABI compilations, each DIE
   still indicates a regx op, rather than a location list
   or an fbreg op, and it is up to us to tie the register
   map to this operation.  Unfortunately, this breaks the
   framework in dwarf2loc.c (see the use of these functions
   there). */
static bool cuda_regnum_hack_pc_valid = false;
static CORE_ADDR cuda_regnum_hack_pc;
static CORE_ADDR cuda_regnum_hack_virt_pc;

void
cuda_regnum_pc_pre_hack (struct frame_info *fi)
{
  uint64_t pc;
  struct gdbarch *gdbarch = get_frame_arch (fi);
  struct regcache *regcache = get_current_regcache ();
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);

  gdb_assert (!cuda_regnum_hack_pc_valid);

  if (!cuda_focus_is_device ())
    return;

  if (fi)
    /* If we have frame information, pull the PC from the frame */
    pc = get_frame_pc (fi);
  else
    /* No frame information - just read the current PC */
    regcache_cooked_read_unsigned (regcache, pc_regnum, (ULONGEST*)&pc);

  cuda_regnum_hack_virt_pc = pc;
  cuda_regnum_hack_pc = cuda_pc_virt_to_phys (pc);
  cuda_regnum_hack_pc_valid = true;
}

void
cuda_regnum_pc_post_hack (void)
{
  if (cuda_regnum_hack_pc_valid)
    cuda_regnum_hack_pc_valid = false;
}

static regmap_t
cuda_get_physical_register (char *reg_name)
{
  uint32_t dev_id, sm_id, wp_id, ln_id;
  uint64_t start_pc, addr, virt_addr;
  kernel_t kernel;
  const char *func_name = NULL;
  module_t module;
  elf_image_t elf_image;
  struct objfile *objfile;
  regmap_t regmap;
  struct symbol *symbol = NULL;

  gdb_assert (cuda_focus_is_device ());
  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, &ln_id);

  /* Get the search parameters */
  addr      = lane_get_pc (dev_id, sm_id, wp_id, ln_id);
  addr      = cuda_regnum_hack_pc_valid ? cuda_regnum_hack_pc : addr;
  virt_addr = lane_get_virtual_pc (dev_id, sm_id, wp_id, ln_id);
  virt_addr = cuda_regnum_hack_pc_valid ? cuda_regnum_hack_virt_pc : virt_addr;
  kernel    = warp_get_kernel (dev_id, sm_id, wp_id);
  start_pc  = kernel_get_virt_code_base (kernel);
  symbol    = find_pc_function ((CORE_ADDR) virt_addr);
  if (symbol)
    func_name = SYMBOL_LINKAGE_NAME (symbol);
  else
    func_name = cuda_find_kernel_name_from_pc (start_pc, false);
  module    = kernel_get_module (kernel);
  elf_image = module_get_elf_image (module);
  objfile   = cuda_elf_image_get_objfile (elf_image);

  /* Do the search */
  regmap = regmap_table_search (objfile, func_name, reg_name, addr);
  
  /* This is a fallback for the cloning=yes case.
     It can be removed once cloning=no is the default. */
  if (regmap_get_num_entries (regmap) == 0)
    {
      func_name = cuda_find_kernel_name_from_pc (start_pc, false);

      /* Do the search */
      regmap = regmap_table_search (objfile, func_name, reg_name, addr);
    }
  
  return regmap;
}

/*
 * Convert a CUDA DWARF register into a physical register index
 */
static int
cuda_reg_to_regnum (struct gdbarch *gdbarch, reg_t reg)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  int max_regs = gdbarch_num_regs (gdbarch) + gdbarch_num_pseudo_regs (gdbarch);
  int32_t regno, decoded_reg;
  uint32_t num_regs;
  char reg_name[8+1];
  regmap_t regmap;

  /* The register is already decoded */
  if (reg >= 0 && reg <= max_regs)
    return reg;

  /* The register is encoded with its register class. */
  if (!cuda_decode_physical_register (reg, &decoded_reg))
    return decoded_reg;

  /* Unrecognized register. */
  if (cuda_get_dwarf_register_string (reg, reg_name, sizeof (reg_name)))
    return -1;

  /* At this point, we know that the register is encoded as PTX register string */
  regmap = cuda_get_physical_register (reg_name);
  num_regs = regmap_get_num_entries (regmap);

  /* If no physical register was found. That means the variable being queried
     is not live at this point. Returns a random register index (R0). */
  if (num_regs == 0)
    {
      warning (("Variable is not live at this point. Value is undetermined."));
      regno = tdep->invalid_lo_regnum;
      return regno;
    }

  /* If we found a single SASS register, then we let cuda-gdb handle it
     normally. */
  if (num_regs == 1 && regmap_get_class (regmap, 0) == REG_CLASS_REG_FULL)
    {
      regno = regmap_get_register (regmap, 0);
      return regno;
    }

  /* Every other situation requires us to store data that cannot be represented
     as a single register index (regno). We keep hold of the data until the
     value is to be fetched. */
  regno = tdep->special_regnum;
  return regno;
}


static void
cuda_pseudo_register_read (struct gdbarch *gdbarch,
                           struct regcache *regcache,
                           int regnum,
                           gdb_byte *buf)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  uint32_t dev, sm, wp, ln;
  uint32_t stack_addr, offset, *p, sz;
  int i, sp_regnum, tmp;
  bool high;
  regmap_t regmap;

  cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

  /* Invalid Register */
  if (regnum == tdep->invalid_lo_regnum ||
      regnum == tdep->invalid_hi_regnum)
    {
      *((uint32_t*)buf) = 0U;
      return;
    }

  /* single SASS register */
  if (regnum != tdep->special_regnum)
    {
      cuda_api_read_register (dev, sm, wp, ln, regnum, (uint32_t*)buf);
      return;
    }

  /* Any combination of SASS register, SP + offset, LMEM offset locations */
  regmap = regmap_get_search_result ();
  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      p = &((uint32_t*)buf)[i];
      sz = sizeof *p;

      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            cuda_api_read_register (dev, sm, wp, ln, regnum, p);
            break;

          case REG_CLASS_MEM_LOCAL:
            gdb_assert (!cuda_current_active_elf_image_uses_abi ());
            offset = regmap_get_offset (regmap, i);
            cuda_api_read_local_memory (dev, sm, wp, ln, offset, p, sz);
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            gdb_assert (cuda_current_active_elf_image_uses_abi ());
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            cuda_api_read_register (dev, sm, wp, ln, sp_regnum, &stack_addr);
            cuda_api_read_local_memory (dev, sm, wp, ln, stack_addr + offset, p, sz);
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            cuda_api_read_register (dev, sm, wp, ln, regnum, &tmp);
            *p = high ? tmp >> 16 : tmp & 0xffff;
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}


static void
cuda_pseudo_register_write (struct gdbarch *gdbarch,
                            struct regcache *regcache,
                            int regnum,
                            const gdb_byte *buf)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  uint32_t dev, sm, wp, ln;
  uint32_t stack_addr, offset, val, sz, old_val, new_val;
  void *ptr;
  int i, sp_regnum;
  bool high;
  regmap_t regmap;

  cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

  /* invalid register */
  if (regnum == tdep->invalid_lo_regnum ||
      regnum == tdep->invalid_hi_regnum)
    {
      error (_("Invalid register."));
      return;
    }

  /* single SASS register */
  if (regnum != tdep->special_regnum)
    {
      val = *(uint32_t*)buf;
      cuda_api_write_register (dev, sm, wp, ln, regnum, val);
      return;
    }

  /* Any combination of SASS register, SP + offset, LMEM offset locations */
  regmap = regmap_get_search_result ();
  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      val = ((uint32_t*)buf)[i];
      ptr = (void*)&((uint32_t*)buf)[i];
      sz = sizeof val;

      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            cuda_api_write_register (dev, sm, wp, ln, regnum, val);
            break;

          case REG_CLASS_MEM_LOCAL:
            gdb_assert (!cuda_current_active_elf_image_uses_abi ());
            offset = regmap_get_offset (regmap, i);
            cuda_api_write_local_memory (dev, sm, wp, ln, offset, ptr, sz);
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            gdb_assert (cuda_current_active_elf_image_uses_abi ());
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            cuda_api_read_register (dev, sm, wp, ln, sp_regnum, &stack_addr);
            cuda_api_write_local_memory (dev, sm, wp, ln, stack_addr + offset, ptr, sz);
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            cuda_api_read_register (dev, sm, wp, ln, regnum, &old_val);
            if (high)
              new_val = (val << 16) | (old_val & 0x0000ffff);
            else
              new_val = (old_val & 0xffff0000) | (val);
            cuda_api_write_register (dev, sm, wp, ln, regnum, new_val);
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}

static void
cuda_print_registers_info (struct gdbarch *gdbarch,
                           struct ui_file *file,
                           struct frame_info *frame,
                           int regnum,
                           int print_all)
{
  int i;
  const int num_regs = gdbarch_num_regs (gdbarch);
  const int device_num_regs = device_get_num_registers (cuda_current_device ());
  gdb_byte buffer[MAX_REGISTER_SIZE];
  const char *register_name = NULL;
  struct value_print_options opts;

  gdb_assert (device_num_regs < num_regs);

  for (i = 0; i < num_regs; i++)
    {
      /* Ignore the registers not specified by the user */
      if (regnum != -1 && regnum != i)
        continue;

      /* Ignore the registers not supported by this device */
      if (i < num_regs && i >= device_num_regs)
        continue;

      /* Print the register name */
      register_name = gdbarch_register_name (gdbarch, i);
      gdb_assert (register_name);
      fputs_filtered (register_name, file);
      print_spaces_filtered (15 - strlen (register_name), file);

      /* Get the data in raw format.  */
      if (!frame_register_read (frame, i, buffer))
      {
        fprintf_filtered (file, "*value not available*\n");
        continue;
      }

      /* Print the register in hexadecimal format */
      get_formatted_print_options (&opts, 'x');
      opts.deref_ref = 1;
      val_print (register_type (gdbarch, i), buffer, 0, 0,
                 file, 0, NULL, &opts, current_language);

      /* Print a tab */
      fprintf_filtered (file, "\t");

      /* Print the register in decimal format */
      get_user_print_options (&opts);
      opts.deref_ref = 1;
      val_print (register_type (gdbarch, i), buffer, 0, 0,
                 file, 0, NULL, &opts, current_language);

      /* Print a newline character */
      fprintf_filtered (file, "\n");
    }
}

void
cuda_update_elf_images ()
{
  uint32_t dev_id, sm_id, wp_id;
  kernel_t kernel;

  // if no kernel running, just return - the image still needs to be active
  // in the event we are processing a 'finish' command and the device kernel
  // has terminated.  we need the function's return type to process its
  // return code (even though cuda kernels can currently only return void).
  // we will clean up the current image upon a subsequent kernel launch,
  // or if we are processing application termination.
  if (!cuda_focus_is_device ())
    return;

  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, NULL);
  kernel = warp_get_kernel (dev_id, sm_id, wp_id);
  kernel_load_elf_images (kernel);
}

static int
cuda_print_insn (bfd_vma pc, disassemble_info *info)
{
  uint32_t inst_size;
  bool is_device_address;
  kernel_t kernel;
  const char * inst;
  uint32_t dev_id, sm_id, wp_id;

  if (!cuda_focus_is_device ())
    return 0;

  /* If this isn't a device address, don't bother */
  cuda_api_is_device_code_address (pc, &is_device_address);
  if (!is_device_address)
    return 0;

  /* decode the instruction at the pc */
  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, NULL);
  kernel = warp_get_kernel (dev_id, sm_id, wp_id);
  inst = kernel_disassemble (kernel, pc, &inst_size);

  if (inst)
    info->fprintf_func (info->stream, "%s", inst);
  else
    info->fprintf_func (info->stream, "Cannot disassemble instruction");

  return inst_size;
}

bool
cuda_is_device_code_address (CORE_ADDR addr)
{
  bool is_cuda_addr = false;
  cuda_api_is_device_code_address ((uint64_t)addr, &is_cuda_addr);
  return is_cuda_addr;
}

/*------------------------------------------------------------------------- */

#define __S__(s)                   #s
#define _STRING_(s)          __S__(s)


/* Returns true if obfd points to a CUDA ELF object file
   (checked by machine type).  Otherwise, returns false. */
bool
cuda_is_bfd_cuda (bfd *obfd)
{
  return (obfd &&
          obfd->tdata.elf_obj_data &&
          obfd->tdata.elf_obj_data->elf_header &&
          obfd->tdata.elf_obj_data->elf_header->e_machine == EM_CUDA);
}


/* Return the CUDA ELF ABI version.  If obfd points to a CUDA ELF
   object file and contains a valid CUDA ELF ABI version, it stores
   the ABI version in abi_version and returns true.  Otherwise, it
   returns false. */
bool
cuda_get_bfd_abi_version (bfd *obfd, unsigned int *abi_version)
{
  if (!cuda_is_bfd_cuda (obfd) || !abi_version)
    return false;
  else
    {
      unsigned int abiv = (elf_elfheader (obfd)->e_ident[EI_ABIVERSION]);
      if (CUDA_ELFOSABIV_16BIT <= abiv && abiv <= CUDA_ELFOSABIV_LATEST)
        {
          *abi_version = abiv;
          return true;
        }
      else
        {
          printf_filtered ("CUDA ELF Image contains unknown ABI version: %d\n", abiv);
          gdb_assert (CUDA_ELFOSABIV_16BIT <= abiv && abiv <= CUDA_ELFOSABIV_LATEST);
        }
    }

  return false;
}

/* Returns true if obfd points to a CUDA ELF object file that was
   compiled against the call frame ABI (the abi version is equal
   to CUDA_ELFOSABIV_ABI).  Returns false otherwise. */
bool
cuda_is_bfd_version_call_abi (bfd *obfd)
{
  unsigned int cuda_abi_version;
  bool is_cuda_abi;

  is_cuda_abi = cuda_get_bfd_abi_version (obfd, &cuda_abi_version);
  return (is_cuda_abi && cuda_abi_version >= CUDA_ELFOSABIV_ABI);
}

bool
cuda_current_active_elf_image_uses_abi (void)
{
  kernel_t    kernel;
  module_t    module;
  elf_image_t elf_image;
  uint32_t    dev_id, sm_id, wp_id;

  if (!get_current_context ())
    return false;

  if (!cuda_focus_is_device ())
    return false;

  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, NULL);
  kernel = warp_get_kernel (dev_id, sm_id, wp_id);
  module    = kernel_get_module (kernel);
  elf_image = module_get_elf_image (module);
  gdb_assert (cuda_elf_image_is_loaded (elf_image));
  return cuda_elf_image_uses_abi (elf_image);
}

/* CUDA - breakpoints */
/* Like breakpoint_address_match, but gdbarch is a parameter. Required to
   evaluate gdbarch_has_global_breakpoints (gdbarch) in the right context. */
int
cuda_breakpoint_address_match (struct gdbarch *gdbarch,
                               struct address_space *aspace1, CORE_ADDR addr1,
                               struct address_space *aspace2, CORE_ADDR addr2)
{
  return ((gdbarch_has_global_breakpoints (gdbarch)
           || aspace1 == aspace2)
          && addr1 == addr2);
}

static CORE_ADDR
cuda_get_symbol_address (char *name)
{
  struct minimal_symbol *sym = lookup_minimal_symbol (name, NULL, NULL);

/* CUDA - Mac OS X specific */
#ifdef target_check_is_objfile_loaded
  struct objfile *objfile;

  /* CUDA - MAC OS X specific
     We need to check that the object file is actually loaded into
     memory, rather than accessing a cached set of symbols. */
  if (sym && sym->ginfo.bfd_section && sym->ginfo.bfd_section->owner)
    {
      objfile = find_objfile_by_name (sym->ginfo.bfd_section->owner->filename, 1); /* 1 = exact match */
      if (objfile && target_check_is_objfile_loaded (objfile))
        return SYMBOL_VALUE_ADDRESS (sym);
    }
#else
  if (sym)
    return SYMBOL_VALUE_ADDRESS (sym);
#endif

  return 0;
}

static void
cuda_signal_handler (int signo)
{
  void *buffer[100];
  int n;

  psignal (signo, "Error: received unexpected signal");

  n = backtrace (buffer, sizeof buffer);
  fprintf (stderr, "BACKTRACE (%d frames):\n", n);
  backtrace_symbols_fd (buffer, n, STDERR_FILENO);
  fflush (stderr);

  cuda_cleanup ();

  exit (1);
}

static void
cuda_signals_initialize (void)
{
  signal (SIGSEGV, cuda_signal_handler);
  signal (SIGPIPE, cuda_signal_handler);
}

void
cuda_cleanup ()
{
  cuda_trace ("cuda_cleanup");

  set_current_context (NULL);
  cuda_system_cleanup_breakpoints ();
  cuda_cleanup_auto_breakpoints (NULL);
  cuda_cleanup_cudart_symbols ();
  cuda_cleanup_tex_maps ();
  cuda_coords_invalidate_current ();
  cuda_system_cleanup_contexts ();
  if (cuda_initialized)
    cuda_system_finalize ();
  cuda_sstep_reset (false);
  cuda_notification_reset ();
  cuda_api_finalize ();

  inferior_in_debug_mode = false;
  cuda_initialized = false;
}

void
cuda_final_cleanup (void *unused)
{
  if (cuda_initialized)
    cuda_api_finalize ();
}

/* Initialize the CUDA debugger API and collect the static data about
   the devices. Once per application run. */
static void
cuda_initialize ()
{
  if (!cuda_initialized)
    {
      cuda_signals_initialize ();
      cuda_api_set_notify_new_event_callback (cuda_notification_notify);
      if (!cuda_api_initialize ())
        {
          cuda_initialized = true;
          cuda_system_initialize ();
        }
    }
}

/* Tell the target application that it is being
   CUDA-debugged. Inferior must have been launched first. */
void
cuda_initialize_target ()
{
  const unsigned char one = 1;
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR gdbPidAddr;
  CORE_ADDR apiClientRevAddr;
  CORE_ADDR sessionIdAddr;
  uint32_t pid;
  uint32_t apiClientRev = CUDBG_API_VERSION_REVISION;
  uint32_t sessionId = cuda_gdb_session_get_id ();

  if (!inferior_in_debug_mode)
    {
      debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
      if (debugFlagAddr)
        {
          target_write_memory (debugFlagAddr, &one, 1);
          rpcFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_RPC_ENABLED));
          pid = getpid ();
          gdbPidAddr = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_PID));
          apiClientRevAddr = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_REVISION));
          sessionIdAddr = cuda_get_symbol_address (_STRING_(CUDBG_SESSION_ID));
          if (rpcFlagAddr && gdbPidAddr && apiClientRevAddr && sessionIdAddr)
            {
              target_write_memory (gdbPidAddr, (char*)&pid, sizeof (pid));
              target_write_memory (rpcFlagAddr, &one, 1);
              target_write_memory (apiClientRevAddr, (char*)&apiClientRev, sizeof(apiClientRev));
              target_write_memory (sessionIdAddr, (char*)&sessionId, sizeof(sessionId));
              inferior_in_debug_mode = true;
            }
          else
            {
              kill (inferior_ptid.lwp, SIGKILL);
              error (_("CUDA application cannot be debugged:  driver incompatibility."));
            }
        }
    }

  cuda_initialize ();
}

bool
cuda_inferior_in_debug_mode (void)
{
  return inferior_in_debug_mode;
}

int
cuda_address_class_type_flags (int byte_size, int addr_class)
{
  switch (addr_class)
    {
      case ptxCodeStorage:    return TYPE_INSTANCE_FLAG_CUDA_CODE;
      case ptxConstStorage:   return TYPE_INSTANCE_FLAG_CUDA_CONST;
      case ptxGenericStorage: return TYPE_INSTANCE_FLAG_CUDA_GENERIC;
      case ptxGlobalStorage:  return TYPE_INSTANCE_FLAG_CUDA_GLOBAL;
      case ptxParamStorage:   return TYPE_INSTANCE_FLAG_CUDA_PARAM;
      case ptxSharedStorage:  return TYPE_INSTANCE_FLAG_CUDA_SHARED;
      case ptxTexStorage:     return TYPE_INSTANCE_FLAG_CUDA_TEX;
      case ptxLocalStorage:   return TYPE_INSTANCE_FLAG_CUDA_LOCAL;
      case ptxRegStorage:     return TYPE_INSTANCE_FLAG_CUDA_REG;
      default:                return 0;
    }
}

static const char *
cuda_address_class_type_flags_to_name (struct gdbarch *gdbarch, int type_flags)
{
  switch (type_flags)
    {
    case TYPE_INSTANCE_FLAG_CUDA_CODE:    return "code";
    case TYPE_INSTANCE_FLAG_CUDA_CONST:   return "constant";
    case TYPE_INSTANCE_FLAG_CUDA_GENERIC: return "generic";
    case TYPE_INSTANCE_FLAG_CUDA_GLOBAL:  return "global";
    case TYPE_INSTANCE_FLAG_CUDA_PARAM:   return "parameter";
    case TYPE_INSTANCE_FLAG_CUDA_SHARED:  return "shared";
    case TYPE_INSTANCE_FLAG_CUDA_TEX:     return "texture";
    case TYPE_INSTANCE_FLAG_CUDA_LOCAL:   return "local";
    case TYPE_INSTANCE_FLAG_CUDA_REG:     return "register";
    default:                              return "unknown_segment";
    }
}

static int
cuda_address_class_name_to_type_flags (struct gdbarch *gdbarch,
                                       const char *name,
                                       int *type_flags)
{
  if (strcmp (name, "code") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_CODE;
      return 1;
    }

  if (strcmp (name, "constant") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_CONST;
      return 1;
    }

  if (strcmp (name, "generic") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_GENERIC;
      return 1;
    }

  if (strcmp (name, "global") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_GLOBAL;
      return 1;
    }

  if (strcmp (name, "parameter") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_PARAM;
      return 1;
    }

  if (strcmp (name, "shared") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_SHARED;
      return 1;
    }

  if (strcmp (name, "texture") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_TEX;
      return 1;
    }

  if (strcmp (name, "register") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_REG;
      return 1;
    }

  if (strcmp (name, "local") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_LOCAL;
      return 1;
    }

  return 0;
}

void
cuda_print_lmem_address_type (void)
{
  struct gdbarch *gdbarch = get_current_arch ();
  const char *lmem_type =
    cuda_address_class_type_flags_to_name (gdbarch,
                                           TYPE_INSTANCE_FLAG_CUDA_LOCAL);

  printf_filtered ("(@%s unsigned *) ", lmem_type);
}

/* Temporary: intercept memory addresses when accessing known
   addresses pointing to CUDA RT variables. Returns 0 if found a CUDA
   RT variable, and 1 otherwise. */
static int
read_cudart_variable (uint64_t address, void * buffer, unsigned amount)
{
  CuDim3 thread_idx, block_dim;
  CuDim3 block_idx, grid_dim;
  uint32_t num_lanes;
  uint32_t dev_id, sm_id, wp_id, ln_id;
  struct {short x,y,z;} short_thread_idx, short_block_dim;
  struct {short x,y,z;} short_block_idx, short_grid_dim;
  kernel_t kernel;

  if (address < CUDBG_BUILTINS_MAX)
    return 1;

  if (!cuda_focus_is_device ())
    return 1;

  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, &ln_id);

  if (CUDBG_THREADIDX_OFFSET <= address)
    {
      thread_idx = lane_get_thread_idx (dev_id, sm_id, wp_id, ln_id);
      short_thread_idx.x = (short) thread_idx.x;
      short_thread_idx.y = (short) thread_idx.y;
      short_thread_idx.z = (short) thread_idx.z;
      memcpy (buffer, (char*)&short_thread_idx +
             (int64_t)address - CUDBG_THREADIDX_OFFSET, amount);
    }
  else if (CUDBG_BLOCKIDX_OFFSET <= address)
    {
      block_idx = warp_get_block_idx (dev_id, sm_id, wp_id);
      short_block_idx.x = (short) block_idx.x;
      short_block_idx.y = (short) block_idx.y;
      short_block_idx.z = (short) block_idx.z;
      memcpy (buffer, (char*)&short_block_idx
             + (int64_t)address - CUDBG_BLOCKIDX_OFFSET, amount);
    }
  else if (CUDBG_BLOCKDIM_OFFSET <= address)
    {
      kernel = warp_get_kernel (dev_id, sm_id, wp_id);
      block_dim = kernel_get_block_dim (kernel);
      short_block_dim.x = (short) block_dim.x;
      short_block_dim.y = (short) block_dim.y;
      short_block_dim.z = (short) block_dim.z;
      memcpy (buffer, (char*)&short_block_dim
             + (int64_t)address - CUDBG_BLOCKDIM_OFFSET, amount);
    }
  else if (CUDBG_GRIDDIM_OFFSET <= address)
    {
      kernel = warp_get_kernel (dev_id, sm_id, wp_id);
      grid_dim = kernel_get_grid_dim (kernel);
      short_grid_dim.x = (short) grid_dim.x;
      short_grid_dim.y = (short) grid_dim.y;
      short_grid_dim.z = (short) grid_dim.z;
      memcpy (buffer, (char*)&short_grid_dim
             + (int64_t)address - CUDBG_GRIDDIM_OFFSET, amount);
    }
  else if (CUDBG_WARPSIZE_OFFSET <= address)
    {
      dev_id    = cuda_current_device ();
      num_lanes = device_get_num_lanes (dev_id);
      memcpy (buffer, &num_lanes, amount);
    }
  else
    return 1;

  return 0;
}

/* Read LEN bytes of CUDA memory at address ADDRESS, placing the
   result in GDB's memory at BUF. Returns 0 on success, and 1
   otherwise. This is used only by partial_memory_read. */
int
cuda_read_memory_partial (CORE_ADDR address, gdb_byte *buf, int len, struct type *type)
{
  int flag;
  uint32_t dev, sm, wp, ln;
  uint32_t tex_id, dim;
  uint32_t *coords;
  bool is_bindless;

  /* No CUDA. Return 1 */
  if (!cuda_debugging_enabled)
    return 1;

  /* Sanity */
  gdb_assert (type);

  /* If address is marked as belonging to a CUDA memory segment, use the
     appropriate API call. */
  flag = TYPE_CUDA_ALL(type);
  if (flag)
    {
      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

      if (TYPE_CUDA_CODE(type))
        cuda_api_read_code_memory (dev, address, buf, len);
      else if (TYPE_CUDA_CONST(type))
        cuda_api_read_const_memory (dev, address, buf, len);
      else if (TYPE_CUDA_GENERIC(type))
        cuda_api_read_global_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_GLOBAL(type))
        cuda_api_read_global_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_PARAM(type))
        cuda_api_read_param_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_SHARED(type))
        cuda_api_read_shared_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_TEX(type))
        {
          cuda_texture_dereference_tex_contents (address, &tex_id, &dim, &coords, &is_bindless);
          if (is_bindless)
            cuda_api_read_texture_memory_bindless (dev, sm, wp, tex_id, dim, coords, buf, len);
          else
            cuda_api_read_texture_memory (dev, sm, wp, tex_id, dim, coords, buf, len);
        }
      else if (TYPE_CUDA_LOCAL(type))
        cuda_api_read_local_memory (dev, sm, wp, ln, address, buf, len);
      else
        error (_("Unknown storage specifier."));
      return 0;
    }

  return 1;
}

/* If there is an address class associated with this value, we've
   stored it in the type.  Check this here, and if set, read from the
   appropriate segment. */
void
cuda_read_memory (CORE_ADDR address, struct value *val, struct type *type, int len)
{
  uint32_t dev, sm, wp, ln;
  gdb_byte *buf = value_contents_all_raw (val);
 
  /* Textures: read tex contents now and dereference the contents on the second
     call to cuda_read_memory. See below. */
  if (IS_TEXTURE_TYPE (type) || cuda_texture_is_tex_ptr (type))
    {
      cuda_texture_read_tex_contents (address, buf);
      return;
    }

  /* No CUDA. Read the host memory */
  if (!cuda_debugging_enabled)
    goto bailout;

  /* Call the partial memory read. Return on success */
  if (cuda_read_memory_partial (address, buf, len, type) == 0)
    return;

   /* The variable is on the stack. It happens when not in the innermost frame. */
  if (value_stack (val) && cuda_focus_is_device ())
   {
      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
      cuda_api_read_local_memory (dev, sm, wp, ln, address, buf, len);
      return;
   }

  /* If address of a built-in CUDA runtime variable, intercept it */
  if (!read_cudart_variable (address, buf, len))
    return;

  /* Default: read the host memory as usual */
bailout:
  if (value_stack (val))
    read_stack (address, buf, len);
  else
    read_memory (address, buf, len);
}

/* FIXME: This is to preserve the symmetry of cuda_read/write_memory_partial. */
int
cuda_write_memory_partial (CORE_ADDR address, const gdb_byte *buf, struct type *type)
{
  int flag;
  uint32_t dev, sm, wp, ln;
  int len = TYPE_LENGTH (type);

  /* No CUDA. Return 1. */
  if (!cuda_debugging_enabled)
    return 1;

  /* If address is marked as belonging to a CUDA memory segment, use the
     appropriate API call. */
  flag = TYPE_CUDA_ALL(type);
  if (flag)
    {
      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

      if (TYPE_CUDA_REG(type))
        {
          /* The following explains how we can come down this path, and why
             cuda_api_write_local_memory is called when the address class
             indicates ptxRegStorage.

             We should only enter this case if we are:
                 1. debugging an application that is using the ABI
                 2. modifying a variable that is mapped to a register that has
                    been saved on the stack
                 3. not modifying a variable for the _innermost_ device frame
                    (as this would follow the cuda_pseudo_register_write path).

             We can possibly add additional checks to ensure that address is
             within the permissable stack range, but cuda_api_write_local_memory
             better return an appropriate error in that case anyway, so let's
             test the API.

             Note there is no corresponding case in cuda_read_memory_with_valtype,
             because _reading_ a previous frame's (saved) registers is all done
             directly by prev register methods (dwarf2-frame.c, cuda-tdep.c).

             As an alternative, we could intercept the value type prior to
             reaching this function and change it to ptxLocalStorage, but that
             can make debugging somewhat difficult. */
          gdb_assert (cuda_current_active_elf_image_uses_abi ());
          cuda_api_write_local_memory (dev, sm, wp, ln, address, buf, len);
        }
      else if (TYPE_CUDA_GENERIC(type))
        cuda_api_write_global_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_GLOBAL(type))
        cuda_api_write_global_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_PARAM(type))
        cuda_api_write_param_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_SHARED(type))
        cuda_api_write_shared_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_LOCAL(type))
        cuda_api_write_local_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_TEX(type))
        error (_("Writing to texture memory is not allowed."));
      else if (TYPE_CUDA_CODE(type))
        error (_("Writing to code memory is not allowed."));
      else if (TYPE_CUDA_CONST(type))
        error (_("Writing to constant memory is not allowed."));
      else
        error (_("Unknown storage specifier."));
      return 0;
    }
  return 1;
}


/* If there is an address class associated with this value, we've
   stored it in the type.  Check this here, and if set, write to the
   appropriate segment. */
void
cuda_write_memory (CORE_ADDR address, const gdb_byte *buf, struct type *type)
{
  int len = TYPE_LENGTH (type);

  /* No CUDA. Write the host memory */
  if (!cuda_debugging_enabled)
    {
      write_memory (address, buf, len);
      return;
    }

  /* Call the partial memory write, return on success */
  if (cuda_write_memory_partial (address, buf, type) == 0)
    return;

  /* Default: write the host memory as usual */
  write_memory (address, buf, len);
}

/* Single-Stepping

   The following data structures and routines are used as a framework to manage
   single-stepping with CUDA devices. It currently solves 2 issues

   1. When single-stepping a warp, we do not want to resume the host if we do
   not have to. The single-stepping framework allows for making GDB believe
   that everything was resumed and that a SIGTRAP was received after each step.

   2. When single-stepping a warp, other warps may be required to be stepped
   too. Out of convenience to the user, we want to keep single-stepping those
   other warps alongside the warp in focus. By doing so, stepping over a
   __syncthreads() instruction will bring all the warps in the same block to
   the next source line.

   This result is achieved by marking the warps we want to single-step with a
   warp mask. When the user issues a new command, the warp is initialized
   accordingly. If the command is a step command, we initialize the warp mask
   with the warp mask and let the mask grow over time as stepping occurs (there
   might be more than one step). If the command is not a step command, the warp
   mask is set empty and will remain that way. In that situation, if
   single-stepping is required, only the minimum number of warps will be
   single-stepped. */

static struct {
  bool     active;
  ptid_t   ptid;
  uint32_t dev_id;
  uint32_t sm_id;
  uint32_t wp_id;
  uint32_t grid_id;
  uint64_t warp_mask;
} cuda_sstep_info;

bool
cuda_sstep_is_active (void)
{
  return cuda_sstep_info.active;
}

ptid_t
cuda_sstep_ptid (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.ptid;
}

void
cuda_sstep_set_ptid (ptid_t ptid)
{
  gdb_assert (cuda_sstep_info.active);
  cuda_sstep_info.ptid = ptid;
}

void
cuda_sstep_execute (ptid_t ptid)
{
  uint32_t dev_id, sm_id, wp_id, grid_id, wp;
  kernel_t kernel;
  uint64_t warp_mask, stepped_warp_mask;
  bool     sstep_other_warps;

  gdb_assert (!cuda_sstep_info.active);
  gdb_assert (cuda_focus_is_device ());

  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, NULL);
  kernel  = warp_get_kernel (dev_id, sm_id, wp_id);
  grid_id = kernel_get_grid_id (kernel);
  sstep_other_warps = cuda_sstep_info.warp_mask != 0ULL;
  stepped_warp_mask = 0ULL;

  if (!sstep_other_warps)
    cuda_sstep_info.warp_mask = (1ULL << wp_id);

  cuda_trace ("device %u sm %u: single-stepping warp mask 0x%"PRIx64"\n",
              dev_id, sm_id, cuda_sstep_info.warp_mask);
  gdb_assert (cuda_sstep_info.warp_mask & (1ULL << wp_id));

  /* Single-step all the warps in the warp mask. */
  for (wp = 0; wp < CUDBG_MAX_WARPS; ++wp)
    if (cuda_sstep_info.warp_mask & (1ULL << wp) &&
        warp_is_valid (dev_id, sm_id, wp))
    {
      warp_single_step (dev_id, sm_id, wp, &warp_mask);
      stepped_warp_mask |= warp_mask;
    }

  /* Update the warp mask. It may have grown. */
  cuda_sstep_info.warp_mask = stepped_warp_mask;

  /* If any warps are marked invalid, but are in the warp_mask
     clear them. This can happen if we stepped a warp over an exit */
  for (wp = 0; wp < CUDBG_MAX_WARPS; ++wp)
    if (cuda_sstep_info.warp_mask & (1ULL << wp) &&
        !warp_is_valid (dev_id, sm_id, wp))
      cuda_sstep_info.warp_mask &= ~(1ULL << wp);

  /* Remember the single-step parameters to trick GDB */
  cuda_sstep_info.active    = true;
  cuda_sstep_info.ptid      = ptid;
  cuda_sstep_info.dev_id    = dev_id;
  cuda_sstep_info.sm_id     = sm_id;
  cuda_sstep_info.wp_id     = wp_id;
  cuda_sstep_info.grid_id   = grid_id;
}

void
cuda_sstep_initialize (bool stepping)
{
  if (stepping && cuda_focus_is_device ())
    cuda_sstep_info.warp_mask = (1ULL << cuda_current_warp ());
  else
    cuda_sstep_info.warp_mask = 0ULL;
}

void
cuda_sstep_reset (bool sstep)
{
/*  When a subroutine is entered while stepping the device, cuda-gdb will
    insert a breakpoint and resume the device. When this happens, the focus
    may change due to the resume. This will cause the cached single step warp
    mask to be incorrect, causing an assertion failure. The fix here is to
    reset the warp mask when switching to a resume. This will cause
    single step execute to update the warp mask after performing the step. */
  if (!sstep && cuda_focus_is_device () && cuda_sstep_is_active ())
    cuda_sstep_info.warp_mask = 0ULL;

  cuda_sstep_info.active = false;
}

bool
cuda_sstep_kernel_has_terminated (void)
{
  uint32_t dev_id, sm_id, wp_id, grid_id;
  cuda_iterator itr;
  cuda_coords_t filter = CUDA_WILDCARD_COORDS;
  bool found_valid_warp;

  gdb_assert (cuda_sstep_info.active);

  dev_id  = cuda_sstep_info.dev_id;
  sm_id   = cuda_sstep_info.sm_id;
  wp_id   = cuda_sstep_info.wp_id;
  grid_id = cuda_sstep_info.grid_id;

  if (warp_is_valid (dev_id, sm_id, wp_id))
    return false;

  found_valid_warp = false;
  filter           = CUDA_WILDCARD_COORDS;
  filter.dev       = dev_id;
  filter.gridId    = grid_id;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_WARPS, &filter, CUDA_SELECT_VALID);
  for (cuda_iterator_start (itr); !cuda_iterator_end (itr); cuda_iterator_next (itr))
    {
      gdb_assert (cuda_iterator_get_current (itr).dev    == dev_id);
      gdb_assert (cuda_iterator_get_current (itr).gridId == grid_id);
      found_valid_warp = true;
      break;
    }
  cuda_iterator_destroy (itr);

  if (found_valid_warp)
    return false;

  return true;
}

/* Frame management */

bool cuda_frame_p (struct frame_info *);
bool cuda_frame_outermost_p (struct frame_info *);
static struct cuda_frame_cache * cuda_frame_cache (struct frame_info *, void **);
static CORE_ADDR cuda_frame_base_address (struct frame_info *, void **);
static void cuda_frame_this_id (struct frame_info *, void **, struct frame_id *);
static struct value * cuda_frame_prev_register (struct frame_info *, void **, int);
static int cuda_frame_sniffer_check (const struct frame_unwind *, struct frame_info *, void **);
static CORE_ADDR cuda_frame_prev_pc (struct frame_info *);

struct cuda_frame_cache
{
  CORE_ADDR base;
  CORE_ADDR pc;
};

static const struct frame_unwind cuda_frame_unwind =
{
  NORMAL_FRAME,
  cuda_frame_this_id,
  cuda_frame_prev_register,
  NULL,
  cuda_frame_sniffer_check,
  NULL,
  NULL,
};

static const struct frame_base cuda_frame_base =
{
  &cuda_frame_unwind,
  cuda_frame_base_address,
  cuda_frame_base_address,
  cuda_frame_base_address
};

/* Returns true if the frame corresponds to a CUDA device function. */
bool
cuda_frame_p (struct frame_info *next_frame)
{
  if (cuda_focus_is_device ())
    return true;
  else
    return false;
}

static bool
cuda_abi_frame_outermost_p (struct frame_info *next_frame)
{
  int call_depth = 0;
  int syscall_call_depth = 0;
  int next_level = 0;
  int this_level = 0;
  cuda_coords_t c;

  /* For ABI compilations, we need to check that this level is equal
     to the call depth.  If so, it's the outermost device frame. */
  cuda_coords_get_current (&c);
  call_depth = lane_get_call_depth (c.dev, c.sm, c.wp, c.ln);

  /* In normal execution, we want to hide syscall frames. This is particularly
     relevant when the application encounters an assertion on the device.
     In this scenario, the call_depth is modified so that the frames
     belonging to the syscall are hidden. */
  if (cuda_options_hide_internal_frames ())
      syscall_call_depth = lane_get_syscall_call_depth (c.dev, c.sm, c.wp, c.ln);

  next_level = frame_relative_level (next_frame);
  this_level = next_level + 1;

  if (this_level >= call_depth - syscall_call_depth)
    return true;

  return false;
}

static bool
cuda_noabi_frame_outermost_p (struct frame_info *next_frame)
{
  int next_level = frame_relative_level (next_frame);

  /* For non-ABI compilations, there is only one device frame which is
     at level 0 (it's next frame is the sentinel at level -1). */
  if (next_level == -1)
    return true;

  return false;
}


/* Returns true if the current frame (next_frame->prev) is the
   outermost device frame. */
bool
cuda_frame_outermost_p (struct frame_info *next_frame)
{
  if (!cuda_frame_p (next_frame))
    return false;

  if (cuda_current_active_elf_image_uses_abi ())
    return cuda_abi_frame_outermost_p (next_frame);
  else
    return cuda_noabi_frame_outermost_p (next_frame);
}

static CORE_ADDR
cuda_abi_frame_cache_base (struct frame_info *next_frame)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  gdb_byte buf[8];

  memset (buf, 0, sizeof (buf));
  frame_unwind_register (next_frame, tdep->sp_regnum, buf);
  return extract_unsigned_integer (buf, sizeof buf, BFD_ENDIAN_LITTLE);
}

static CORE_ADDR
cuda_noabi_frame_cache_base (struct frame_info *next_frame)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  gdb_byte buf[8];

  if (cuda_frame_outermost_p (next_frame))
      return 0;
  else
    {
      memset (buf, 0, sizeof (buf));
      frame_unwind_register (next_frame, 0 /* dummy */, buf);
      return extract_unsigned_integer (buf, sizeof buf, BFD_ENDIAN_LITTLE);
    }
}

static struct cuda_frame_cache *
cuda_frame_cache (struct frame_info *next_frame, void **this_cache)
{
  struct cuda_frame_cache *cache;

  gdb_assert (cuda_frame_p (next_frame));

  if (*this_cache)
    return *this_cache;

  cache = FRAME_OBSTACK_ZALLOC (struct cuda_frame_cache);
  *this_cache = cache;

  cache->pc = get_frame_func (next_frame);

  if (cuda_current_active_elf_image_uses_abi ())
    cache->base = cuda_abi_frame_cache_base (next_frame);
  else
    cache->base = cuda_noabi_frame_cache_base (next_frame);

  return cache;
}

/* cuda_frame_base_address is not ABI-dependent, since it only
   queries the base field from the frame cache.  It is the frame
   cache itself which is constructed uniquely for ABI/non-ABI
   compilations. */
static CORE_ADDR
cuda_frame_base_address (struct frame_info *next_frame, void **this_cache)
{
  struct cuda_frame_cache *cache;
  CORE_ADDR base;

  gdb_assert (cuda_frame_p (next_frame));

  if (*this_cache)
    return ((struct cuda_frame_cache *)(*this_cache))->base;

  cache = cuda_frame_cache (next_frame, this_cache);
  base = cache->base;
  return base;
}

static struct frame_id
cuda_abi_frame_id_build (struct frame_info *next_frame, void **this_cache,
                         struct frame_id *this_id)
{
  struct cuda_frame_cache *cache;
  int call_depth = 0;
  int syscall_call_depth = 0;
  int next_level;
  cuda_coords_t c;

  cache = cuda_frame_cache (next_frame, this_cache);
  next_level = frame_relative_level (next_frame);

  cuda_coords_get_current (&c);
  call_depth = lane_get_call_depth (c.dev, c.sm, c.wp, c.ln);
  syscall_call_depth = lane_get_syscall_call_depth (c.dev, c.sm, c.wp, c.ln);

  /* With the ABI, we can have multiple device frames. */
  if (next_level < call_depth)
    {
      /* When we have syscall frames, we will build them as special frames,
         as the API will always return only the PC to the first non syscall
         frame. Thus all frames less the syscall_call_depth will be identical
         to the frame at the syscall call depth */
      if (next_level <= syscall_call_depth)
        return frame_id_build_special (cache->base, cache->pc,
                                       syscall_call_depth + next_level);
      else
        return frame_id_build (cache->base, cache->pc);
    }
  else
    return frame_id_build_special (cache->base, cache->pc, 1);
}

static struct frame_id
cuda_noabi_frame_id_build (struct frame_info *next_frame, void **this_cache,
                           struct frame_id *this_id)
{
  struct cuda_frame_cache *cache;
  int next_level = frame_relative_level (next_frame);

  cache = cuda_frame_cache (next_frame, this_cache);

  /* Without the ABI, we have a single device frame.  Device frame and
     dummy frame are identical, except for the dummy frame that has the
     special_p bit set. Required to fool GDB into thinking those are not
     duplicates. Also it is a nice way to differentiate them. */
  if (next_level == -1)
    return frame_id_build_special (cache->base, cache->pc, 0);
  else
    return frame_id_build_special (cache->base, cache->pc, 1);
}

static void
cuda_frame_this_id (struct frame_info *next_frame, void **this_cache,
                    struct frame_id *this_id)
{
  int next_level = frame_relative_level (next_frame);

  gdb_assert (cuda_frame_p (next_frame));

  if (cuda_current_active_elf_image_uses_abi ())
      *this_id = cuda_abi_frame_id_build (next_frame, this_cache, this_id);
  else
      *this_id = cuda_noabi_frame_id_build (next_frame, this_cache, this_id);

  if (frame_debug)
    {
      fprintf_unfiltered (gdb_stdlog, "{ cuda_frame_this_id "
                          "(frame=%d)", next_level);
      fprintf_unfiltered (gdb_stdlog, " -> this_id=");
      fprint_frame_id (gdb_stdlog, *this_id);
      fprintf_unfiltered (gdb_stdlog, " }\n");
    }
}

/* When unwinding registers stored on CUDA ABI frames, we use
   this function to hook in the dwarf2 frame unwind routines. */
static struct value *
cuda_abi_hook_dwarf2_frame_prev_register (struct frame_info *next_frame,
                                          void **this_cache,
                                          int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  void *dwarfcache = NULL;
  struct frame_base *dwarf2_base_finder = NULL;
  struct value *value = NULL;
  gdb_byte *value_contents = NULL;
  CORE_ADDR sp = 0;
  int dwarf2 = 0;

  /* If we have a dwarf2 base finder, then we will use it to know what the
     value of the stack pointer is.  See dwarf2-frame.c */
  if (regnum == tdep->sp_regnum)
    {
      dwarf2_base_finder = (struct frame_base *)dwarf2_frame_base_sniffer (next_frame);
      if (dwarf2_base_finder)
        {
          sp = dwarf2_base_finder->this_base (next_frame, &dwarfcache);
          value = frame_unwind_got_address (next_frame, regnum, sp);
        }
    }

  /* If we have a dwarf2 unwinder, then we will use it to know where to look
     for the value of all CUDA registers.  See dwarf2-frame.c */
  dwarf2 = dwarf2_frame_unwind.sniffer (&dwarf2_frame_unwind,
                                        next_frame,
                                        (void**)&dwarfcache);
  if (!value && dwarf2)
    value = dwarf2_frame_unwind.prev_register (next_frame,
                                               (void **)&dwarfcache,
                                               regnum);

  return value;
}

/* With the ABI, prev_register needs assistance from the dwarf2 frame
   unwinder to decode the storage location of a given regnum for a
   given frame.  Non-debug compilations will not have this assistance,
   so we check for a proper dwarf2 unwinder to make sure.  The PC can
   be decoded without dwarf2 assistance thanks to the device's runtime
   stack. */
static struct value *
cuda_abi_frame_prev_register (struct frame_info *next_frame,
                              void **this_cache,
                              int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  CORE_ADDR pc = 0;
  struct value *value;

  if (regnum == pc_regnum)
    {
      pc = cuda_frame_prev_pc (next_frame);
      value = frame_unwind_got_address (next_frame, regnum, pc);
    }
  else if (frame_relative_level (next_frame) == -1)
    value = frame_unwind_got_register (next_frame, regnum, regnum);
  else
    value = cuda_abi_hook_dwarf2_frame_prev_register (next_frame, this_cache, regnum);

  /* Last resort: if no value found, use the register for the innermost frame. */
  if (!value)
    value = frame_unwind_got_register (next_frame, regnum, regnum);

  return value;
}

/* Without the ABI, prev_register only needs to read current values from
   the register file (with the exception of PC, which requires special
   handling for inserted dummy frames) */
static struct value *
cuda_noabi_frame_prev_register (struct frame_info *next_frame,
                                void **this_cache,
                                int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  struct regcache *regcache = get_current_regcache ();
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  CORE_ADDR pc = 0;

  if (regnum == pc_regnum)
    {
      pc = cuda_frame_prev_pc (next_frame);
      return frame_unwind_got_address (next_frame, regnum, pc);
    }
  else
    return frame_unwind_got_register (next_frame, regnum, regnum);
}

static struct value *
cuda_frame_prev_register (struct frame_info *next_frame,
                          void **this_cache,
                          int regnum)
{
  int next_level = frame_relative_level (next_frame);
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  struct value *value = NULL;

  if (cuda_current_active_elf_image_uses_abi ())
    value = cuda_abi_frame_prev_register (next_frame, this_cache, regnum);
  else
    value = cuda_noabi_frame_prev_register (next_frame, this_cache, regnum);

  if (frame_debug)
    {
      fprintf_unfiltered (gdb_stdlog, "{ cuda_frame_prev_register "
                          "(frame=%d,regnum=%d(%s),...) ",
                          next_level, regnum,
                          user_reg_map_regnum_to_name (gdbarch, regnum));
      fprintf_unfiltered (gdb_stdlog, "->");
      fprintf_unfiltered (gdb_stdlog, " *bufferp=");
      if (value == NULL)
        fprintf_unfiltered (gdb_stdlog, "<NULL>");
      else
        {
          int i;
          const unsigned char *buf = value_contents (value);
          fprintf_unfiltered (gdb_stdlog, "[");
          for (i = 0; i < register_size (gdbarch, regnum); i++)
            fprintf_unfiltered (gdb_stdlog, "%02x", buf[i]);
          fprintf_unfiltered (gdb_stdlog, "]");
        }
      fprintf_unfiltered (gdb_stdlog, " }\n");
    }

  return value;
}

/* cuda_frame_sniffer_check is not ABI-dependent at the moment.
   Ideally, there will be 2 separate sniffers, and we can remove
   switching internally within each of the frame functions. */
static int
cuda_frame_sniffer_check (const struct frame_unwind *self,
                          struct frame_info *next_frame,
                          void **this_prologue_cache)
{
  bool is_cuda_frame;
  int next_level = frame_relative_level (next_frame);

  is_cuda_frame = cuda_frame_p (next_frame) &&
                  self == &cuda_frame_unwind;

  if (frame_debug)
    fprintf_unfiltered (gdb_stdlog, "{ cuda_frame_sniffer_check "
                        "(frame = %d) -> %d }\n", next_level, is_cuda_frame);
  return is_cuda_frame;
}

static CORE_ADDR
cuda_abi_frame_prev_pc (struct frame_info *next_frame)
{
  uint64_t pc;
  int call_depth = 0;
  int syscall_call_depth = 0;
  int next_level;
  cuda_coords_t c;

  next_level = frame_relative_level (next_frame);

  cuda_coords_get_current (&c);
  call_depth = lane_get_call_depth (c.dev, c.sm, c.wp, c.ln);

  if (next_level == -1)
    pc = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
  else if (next_level < call_depth)
    {
      if (cuda_options_hide_internal_frames ())
        {
          syscall_call_depth = lane_get_syscall_call_depth (c.dev, c.sm, c.wp, c.ln);
          if (next_level < syscall_call_depth)
            next_level += syscall_call_depth;
        }
      pc = lane_get_virtual_return_address (c.dev, c.sm, c.wp, c.ln, next_level);
    }
  else
    pc = 0ULL;

  return (CORE_ADDR) pc;
}

static CORE_ADDR
cuda_noabi_frame_prev_pc (struct frame_info *next_frame)
{
  uint64_t pc;
  int next_level = frame_relative_level (next_frame);
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_frame_arch (next_frame);

  cuda_coords_get_current (&c);
  pc = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);

  /* dummy frame PC will point to the kernel entry + 1. Because it is
     not the innermost frame, GDB sees it as the PC after the function
     call and will decrement it by 1 when printing the stack. Therefore
     we increment it by 1 now to make sure the final PC is still within
     the kernel code block. */
  if (next_level == 0)
    pc = get_pc_function_start (pc) + 1;

  return (CORE_ADDR) pc;
}

static CORE_ADDR
cuda_frame_prev_pc (struct frame_info *next_frame)
{
  gdb_assert (cuda_frame_p (next_frame));

  if (cuda_current_active_elf_image_uses_abi ())
    return cuda_abi_frame_prev_pc (next_frame);
  else
    return cuda_noabi_frame_prev_pc (next_frame);
}

const struct frame_base *
cuda_frame_base_sniffer (struct frame_info *next_frame)
{
  const struct frame_base *base = NULL;
  int next_level = frame_relative_level (next_frame);

  if (cuda_frame_p (next_frame))
      base = &cuda_frame_base;

  if (frame_debug)
    fprintf_unfiltered (gdb_stdlog, "{ cuda_frame_base_sniffer "
                        "(frame=%d) -> %d }\n", next_level, !!base);

  return base;
}

const struct frame_unwind *
cuda_frame_sniffer (struct frame_info *next_frame)
{
  const struct frame_unwind *unwind = NULL;
  int next_level = frame_relative_level (next_frame);

  if (cuda_frame_p (next_frame))
    unwind = &cuda_frame_unwind;

  if (frame_debug)
    fprintf_unfiltered (gdb_stdlog, "{ cuda_frame_sniffer (frame=%d) -> %d }\n",
                        next_level, !!unwind);

  return unwind;
}

/* A CUDA internal frame is any frame deeper that the runtime frame (included)
   that is not also a device frame. */
int
cuda_frame_is_internal (struct frame_info *fi)
{
  struct cuda_frame_info *cfi;
  struct frame_info *frame;

  if (!cuda_debugging_enabled || !fi)
    return false;

  cfi = cuda_get_frame_info (fi);
  if (cfi->cuda_internal_p)
    return cfi->cuda_internal;

  if (cuda_frame_p (get_next_frame (fi)))
    {
      cfi->cuda_internal_p = true;
      cfi->cuda_internal = false;
      return false;
    }

  for (frame = fi; frame; frame = get_prev_frame (frame))
    {
      if (cuda_frame_is_runtime_entrypoint (frame))
        {
          cfi->cuda_internal_p = true;
          cfi->cuda_internal = true;
          return true;
        }
    }

  cfi->cuda_internal_p = true;
  cfi->cuda_internal = false;
  return false;
}

/* A CUDA device syscall frame is any frame within the syscall_call_depth */
int
cuda_frame_is_device_syscall (struct frame_info *fi)
{
  struct cuda_frame_info *cfi;
  cuda_coords_t c;
  int syscall_call_depth = 0;
  int call_depth = 0;
  int this_level = 0;

  if (!cuda_debugging_enabled || !fi)
    return false;

  cfi = cuda_get_frame_info (fi);
  if (cfi->cuda_device_syscall_p)
    return cfi->cuda_device_syscall;

  cuda_coords_get_current (&c);
  call_depth = lane_get_call_depth (c.dev, c.sm, c.wp, c.ln);
  syscall_call_depth = lane_get_syscall_call_depth (c.dev, c.sm, c.wp, c.ln);

  this_level = frame_relative_level (fi);
  if (frame_debug)
    {
      fprintf_unfiltered (gdb_stdlog, "this_level:%d syscall_level:%d",
                          this_level, syscall_call_depth);
      fprintf_unfiltered (gdb_stdlog, "->");
    }

  if (this_level <= call_depth && this_level < syscall_call_depth)
    {
      cfi->cuda_device_syscall_p = true;
      cfi->cuda_device_syscall = true;
      return true;
    }

  cfi->cuda_device_syscall_p = true;
  cfi->cuda_device_syscall = false;
  return false;
}

/* The CUDA device frame is the kernel entry point function. */
int
cuda_frame_is_device (struct frame_info *fi)
{
  struct cuda_frame_info *cfi;

  if (!fi)
    return false;

  cfi = cuda_get_frame_info (fi);
  if (cfi->cuda_device_p)
    return cfi->cuda_device;

  cfi->cuda_device_p = true;
  cfi->cuda_device = cuda_frame_p (get_next_frame (fi));
  return cfi->cuda_device;
}

/* The CUDA runtime frame is the non-device frame whose function name
   is the kernel name. */
int
cuda_frame_is_runtime_entrypoint (struct frame_info *fi)
{
  struct cuda_frame_info *cfi;
  char *function_name = NULL;
  uint64_t addr = 0;

  if (!fi)
    return false;

  cfi = cuda_get_frame_info (fi);
  if (cfi->cuda_runtime_entrypoint_p)
    return cfi->cuda_runtime_entrypoint;

  if (cuda_frame_is_device (fi))
    {
      cfi->cuda_runtime_entrypoint_p = true;
      cfi->cuda_runtime_entrypoint = false;
      return false;
    }

  function_name = cuda_find_kernel_name_from_pc (get_frame_address_in_block (fi), false);
  if (function_name && cuda_api_lookup_device_code_symbol (function_name, &addr))
    {
      cfi->cuda_runtime_entrypoint_p = true;
      cfi->cuda_runtime_entrypoint = true;
      return true;
    }

  cfi->cuda_runtime_entrypoint_p = true;
  cfi->cuda_runtime_entrypoint = false;
  return false;
}

/* CUDA: frame relative level is computed recursively to handle hiding
   of internal CUDA runtime frames.

   NOTE: this function needs to co-exist with frame_relative_level to
   avoid a deadlock. cuda_frame_is_internal_frame () needs
   frame_relative_level () to compute its result, and
   cuda_frame_relative_level () needs cuda_frame_is_internal_frame () to
   compute its result... */
int
cuda_frame_relative_level (struct frame_info *fi)
{
  struct cuda_frame_info *cfi;
  int ignored = 0;

  if (fi == NULL)
    return -1;

  cfi = cuda_get_frame_info (fi);
  if (cfi->cuda_level_p)
    return cfi->cuda_level;

  /* If hiding is inactive, no need for recursion. */
  if (!cuda_options_hide_internal_frames ())
    {
      cfi->cuda_level_p = true;
      cfi->cuda_level = frame_relative_level (fi);
      return cfi->cuda_level;
    }

  /* Stop the recursion */
  if (frame_relative_level (fi) == -1)
    {
      cfi->cuda_level_p = true;
      cfi->cuda_level = -1;
      return cfi->cuda_level;
    }

  /* Do not count internal frames. */
  if (cuda_frame_is_internal (fi) || cuda_frame_is_device_syscall (fi))
    {
      cfi->cuda_level_p = true;
      cfi->cuda_level = cuda_frame_relative_level (get_next_frame (fi));
      return cfi->cuda_level;
    }

  cfi->cuda_level_p = true;
  cfi->cuda_level = cuda_frame_relative_level (get_next_frame (fi)) + 1;
  return cfi->cuda_level;
}

/* CUDA ABI return value convention:

   (size == 32-bits) .s32/.u32/.f32/.b32      -> R4
   (size == 64-bits) .s64/.u64/.f64/.b64      -> R4-R5
   (size <= 384-bits) .align N .b8 name[size] -> size <= 384-bits -> R4-R15 (A)
   (size > 384-bits)  .align N .b8 name[size] -> size > 384-bits  -> Memory (B)

   For array case (B), the pointer to the memory location is passed as
   a parameter at the beginning of the parameter list.  Memory is allocated
   in the _calling_ function, which is the consumer of the return value.
*/
static enum return_value_convention
cuda_abi_return_value (struct gdbarch *gdbarch, struct type *func_type,
                       struct type *type, struct regcache *regcache,
                       gdb_byte *readbuf, const gdb_byte *writebuf)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  int regnum = tdep->first_rv_regnum;
  int len = TYPE_LENGTH (type);
  int i;
  uint32_t regval32 = 0U;
  ULONGEST regval   = 0ULL;
  ULONGEST addr;
  uint32_t dev, sm, wp, ln;

  /* The return value is in one or more registers. */
  if (len <= tdep->max_reg_rv_size)
    {
      /* Read/write all regs until we've satisfied len. */
      for (i = 0; len > 0; i++, regnum++, len -= 4)
        {
          if (readbuf)
            {
              regcache_cooked_read_unsigned (regcache, regnum, &regval);
              regval32 = (uint32_t) regval;
              memcpy (readbuf + i * 4, &regval32, min (len, 4));
            }
          if (writebuf)
            {
              memcpy (&regval32, writebuf + i * 4, min (len, 4));
              regval = regval32;
              regcache_cooked_write_unsigned (regcache, regnum, regval);
            }
        }

      return RETURN_VALUE_REGISTER_CONVENTION;
    }

  /* The return value is in memory. */
  if (readbuf)
  {

    /* In the case of large return values, space has been allocated in memory
       to hold the value, and a pointer to that allocation is at the beginning
       of the parameter list.  We need to read the register that holds the
       address, and then read from that address to obtain the value. */
    cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
    regcache_cooked_read_unsigned (regcache, regnum, &addr);
    cuda_api_read_local_memory (dev, sm, wp, ln, addr, readbuf, len);
  }

  return RETURN_VALUE_ABI_RETURNS_ADDRESS;
}

static int
cuda_adjust_regnum (struct gdbarch *gdbarch, int regnum, int eh_frame_p)
{
  int adjusted_regnum = 0;

  gdb_assert (!eh_frame_p);

  /* If not a device register, nothing to adjust. This happens only when called
     by the DWARF2 frame sniffer when determining the type of the frame. It is
     then safe to bail out and not pass the request to the host adjust_regnum
     function, because, at that point, the type of the frame is not yet
     determinted. */
  if (!cuda_focus_is_device ())
    return regnum;

  cuda_decode_physical_register (regnum, &adjusted_regnum);

  return adjusted_regnum;
}

static CORE_ADDR
cuda_skip_prologue (struct gdbarch *gdbarch, CORE_ADDR pc)
{
  CORE_ADDR start_addr, end_addr, post_prologue_pc;

  /* CUDA - skip prologue - temporary
     Until we always have a prologue, even if empty.
     If Tesla kernel generated with open64, there is no prologue */
  {
    struct obj_section *osect = find_pc_section (pc);

    if (osect &&
        osect->objfile &&
        !cuda_is_bfd_version_call_abi (osect->objfile->obfd) &&
        osect->objfile->cuda_objfile &&
        osect->objfile->cuda_producer_is_open64)
      return pc;
  }

  /* See if we can determine the end of the prologue via the symbol table.
     If so, then return either PC, or the PC after the prologue, whichever
     is greater.  */
  if (find_pc_partial_function (pc, NULL, &start_addr, &end_addr))
    {
      post_prologue_pc = skip_prologue_using_sal (gdbarch, start_addr);

      /* There is a bug in skip_prologue_using_sal(). The end PC returned by
         find_pc_sect_line() is off by one instruction. It's pointing to the
         first instruction of the next line instead of the last instruction of
         the current line. I cannot fix it there since the instruction size is
         unknown. But I can fix it here, which also has the advantage of not
         impacting the way gdb behaves with the host code. When that happens,
         it means that the function body is empty (foo(){};). In that case, we
         follow GDB policy and do not skip the prologue. It also allow us to no
         point to the last instruction of a device function. That instruction
         is not guaranteed to be ever executed, which makes setting breakpoints
         trickier. */
      if (post_prologue_pc > end_addr)
        post_prologue_pc = pc;

      /* If the post_prologue_pc does not make sense, return the given PC. */
      if (post_prologue_pc < pc)
        post_prologue_pc = pc;

      return post_prologue_pc;

      /* If we can't adjust the prologue from the symbol table, we may need
         to resort to instruction scanning.  For now, assume the entry above. */
    }

  /* If we can do no better than the original pc, then just return it. */
  return pc;
}

/* CUDA:  Determine whether or not the subprogram DIE requires
   a base address that should be added to all child DIEs.  This
   is determined based on the ABI version in the CUDA ELF image. */
CORE_ADDR
cuda_dwarf2_func_baseaddr (struct objfile *objfile, char *func_name)
{
  bool is_cuda_abi;
  unsigned int cuda_abi_version;

  /* See if this is a CUDA ELF object file and get its abi version.  Before
     CUDA_ELFOSABIV_RELOC, all DWARF DIE low/high PC attributes did not use
     relocators and were 0-based.  After CUDA_ELFOSABIV_RELOC, the low/high
     PC attributes use relocators and are no longer 0-based. */
  is_cuda_abi = cuda_get_bfd_abi_version (objfile->obfd, &cuda_abi_version);
  if (is_cuda_abi && cuda_abi_version < CUDA_ELFOSABIV_RELOC) /* Not relocated */
    {
      if (func_name)
        {
          CORE_ADDR vma;
          if (cuda_find_func_text_vma_from_objfile (objfile, func_name, &vma))
            /* Return section base addr (vma) for this function */
            return vma;
        }
    }

  /* No base address */
  return 0;
}

/* Given a raw function name (string), find if there is a function for it.
   If so, return the function's base (prologue adjusted, if necessary)
   in func_addr.  Returns true if the function is found, false otherwise. */
bool
cuda_find_function_pc_from_objfile (struct objfile *objfile,
                                    char *func_name,
                                    CORE_ADDR *func_addr)
{
  CORE_ADDR addr = 0;
  bool found = false;
  struct block *b;
  struct blockvector *bv;
  struct symtab *s = NULL;
  struct minimal_symbol *msymbol = NULL;
  char *sym_name = NULL;
  char *tmp_func_name = NULL;
  struct gdbarch *gdbarch = get_current_arch ();

  gdb_assert (objfile);
  gdb_assert (func_name);
  gdb_assert (func_addr);

  if (!cuda_is_bfd_cuda (objfile->obfd))
    return false;

  /* Test if a colon exists in the function name string.
     If so, then we need to ignore everything before it
     so that we can search using the function name directly. */
  if ((tmp_func_name = strrchr (func_name, ':')))
    func_name = tmp_func_name + 1;

  /* We need to find the fully-qualified symbol name that func_name
     corresponds to (if any).  This will handle mangled symbol names,
     which is what will be used to lookup a CUDA device code symbol. */
  if ((msymbol = lookup_minimal_symbol (func_name, NULL, objfile)))
    sym_name = msymbol->ginfo.name;

  if (!sym_name)
    return false;

  /* Look for functions - assigned from DWARF, this path will only
     find information for debug compilations. */
  ALL_OBJFILE_SYMTABS (objfile, s)
    {
      int i;
      bv = BLOCKVECTOR (s);
      for (i = 0; i < BLOCKVECTOR_NBLOCKS (bv); i++)
        {
          b = BLOCKVECTOR_BLOCK (bv, i);
          if (!b || !b->function)
            continue;

          if (!SYMBOL_MATCHES_NATURAL_NAME (b->function, sym_name))
            continue;

          found = true;
          addr = BLOCK_START (b);
          addr = cuda_skip_prologue (gdbarch, addr);
          break;
        }

      if (found)
        break;
    }

  /* If we didn't find a function, then it could be a non-debug
     compilation, so look at the msymtab. */
  if (!found)
    {
      ALL_OBJFILE_MSYMBOLS (objfile, msymbol)
        {
          if (!strcmp (sym_name, msymbol->ginfo.name))
            {
              found = true;
              addr = msymbol->ginfo.value.address;
            }
        }
    }

  /* If we found the symbol, return its address in func_addr
     and return true. */
  if (found)
    {
      *func_addr = addr;
      return true;
    }

  return false;
}


/* Given a raw function name (string), find its corresponding text section vma.
   Returns true if found and stores the address in vma.  Returns false otherwise. */
bool
cuda_find_func_text_vma_from_objfile (struct objfile *objfile,
                                      char *func_name,
                                      CORE_ADDR *vma)
{
  struct obj_section *osect = NULL;
  asection *section = NULL;
  char *text_seg_name = NULL;

  gdb_assert (objfile);
  gdb_assert (func_name);
  gdb_assert (vma);

  /* Construct CUDA text segment name */
  text_seg_name = (char *) xmalloc (strlen (CUDA_ELF_TEXT_PREFIX) + strlen (func_name) + 1);
  strcpy (text_seg_name, CUDA_ELF_TEXT_PREFIX);
  strcat (text_seg_name, func_name);

  ALL_OBJFILE_OSECTIONS (objfile, osect)
    {
      section = osect->the_bfd_section;
      if (section)
        {
          if (!strcmp (section->name, text_seg_name))
            {
              /* Found - store address in vma */
              xfree (text_seg_name);
              *vma = section->vma;
              return true;
            }
        }
    }

  /* Not found */
  xfree (text_seg_name);
  return false;
}

/* expensive: traverse all the objfiles to see if addr corresponds to
   any of them. Return the context if found, 0 otherwise. */
context_t
cuda_find_context_by_addr (CORE_ADDR addr)
{
  uint32_t  dev_id;
  context_t context;

  for (dev_id = 0; dev_id < CUDBG_MAX_DEVICES; ++dev_id)
    {
      context = device_find_context_by_addr (dev_id, addr);
      if (context)
        return context;
    }

  return NULL;
}

struct value *
cuda_value_of_builtin_frame_phys_pc_reg (struct frame_info *frame)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);
  struct type *type_data_ptr = builtin_type (gdbarch)->builtin_data_ptr;
  struct value *val = allocate_value (type_data_ptr);
  gdb_byte *buf = value_contents_raw (val);

  if (frame == NULL || !cuda_focus_is_device ())
    memset (buf, 0, TYPE_LENGTH (value_type (val)));
  else
    gdbarch_address_to_pointer (gdbarch, type_data_ptr, buf,
                                cuda_pc_virt_to_phys (get_frame_pc (frame)));
  return val;
}


/* Returns 1 when a value is stored in more than one register (long, double).
   Works with assumption that the compiler allocates consecutive registers for
   those cases.  */
static int
cuda_convert_register_p (struct gdbarch *gdbarch, int regnum, struct type *type)
{
  if (regnum == cuda_pc_regnum (gdbarch))
    return 1;

  return 0;
}

static void
cuda_special_register_to_value (struct frame_info *frame, int regnum,
                                struct type *type, gdb_byte *to)
{
  int i = 0;
  bool high = false;
  regmap_t regmap = NULL;
  uint32_t *p = (uint32_t *)to;
  uint32_t dev = 0, sm = 0, wp = 0, ln = 0;
  uint32_t sp_regnum = 0, offset = 0, stack_addr = 0, val32 = 0;
  struct value *val = NULL;
  struct gdbarch *gdbarch = get_frame_arch (frame);

  gdb_assert (regnum == cuda_special_regnum (gdbarch));

  cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
  regmap = regmap_get_search_result ();

  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            get_frame_register (frame, regnum, (gdb_byte*)&p[i]);
            break;

          case REG_CLASS_MEM_LOCAL:
            offset = regmap_get_offset (regmap, i);
            cuda_api_read_local_memory (dev, sm, wp, ln, offset,
                                        (void*)&p[i], sizeof (p[i]));
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            get_frame_register (frame, sp_regnum, (gdb_byte*)&stack_addr);
            cuda_api_read_local_memory (dev, sm, wp, ln, stack_addr + offset,
                                        (void*)&p[i], sizeof (p[i]));
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            get_frame_register (frame, regnum, (gdb_byte*)&val32);
            p[i] = high ? val32 >> 16 : val32 & 0xffff;
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}

static void
cuda_value_to_special_register (struct frame_info *frame, int regnum,
                                struct type *type, const gdb_byte *from)
{
  int i = 0;
  bool high = false;
  regmap_t regmap = NULL;
  uint32_t *p = (uint32_t *)from;
  uint32_t dev = 0, sm = 0, wp = 0, ln = 0;
  uint32_t sp_regnum = 0, offset = 0, stack_addr = 0, val32 = 0;
  struct value *val = NULL;
  struct gdbarch *gdbarch = get_frame_arch (frame);

  gdb_assert (regnum == cuda_special_regnum (gdbarch));

  cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
  regmap = regmap_get_search_result ();

  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            put_frame_register (frame, regnum, (gdb_byte*)&p[i]);
            break;

          case REG_CLASS_MEM_LOCAL:
            offset = regmap_get_offset (regmap, i);
            cuda_api_write_local_memory (dev, sm, wp, ln, offset,
                                        (void*)&p[i], sizeof (p[i]));
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            get_frame_register (frame, sp_regnum, (gdb_byte*)&stack_addr);
            cuda_api_write_local_memory (dev, sm, wp, ln, stack_addr + offset,
                                         (void*)&p[i], sizeof (p[i]));
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            get_frame_register (frame, regnum, (gdb_byte*)&val32);
            val32 = high ? (val32 & 0xffff)     | (p[i] << 16)
                         : (val32 & 0xffff0000) | (p[i] & 0xffff);
            put_frame_register (frame, regnum, (gdb_byte*)&val32);
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}

/* Read a value of type TYPE from register REGNUM in frame FRAME, and
   return its contents in TO.  */
static void
cuda_register_to_value (struct frame_info *frame, int regnum,
                        struct type *type, gdb_byte *to)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);

  if (regnum == cuda_special_regnum (gdbarch))
    cuda_special_register_to_value (frame, regnum, type, to);
  else if (regnum == cuda_pc_regnum (gdbarch))
    gdb_assert (0); // use cuda_frame_prev_pc
  else
    get_frame_register (frame, regnum, to);
}

/* Write the contents FROM of a value of type TYPE into register
   REGNUM in frame FRAME.  */
static void
cuda_value_to_register (struct frame_info *frame, int regnum,
                        struct type *type, const gdb_byte *from)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);

  if (regnum == cuda_special_regnum (gdbarch))
    cuda_value_to_special_register (frame, regnum, type, from);
  else if (regnum == cuda_pc_regnum (gdbarch))
    gdb_assert (0); // cannot write PC
  else
    put_frame_register (frame, regnum, from);
}

static struct value*
cuda_value_from_register (struct type *type, int regnum, struct frame_info *frame)
{
  int level = frame_relative_level (frame);
  struct gdbarch *gdbarch = get_frame_arch (frame);
  struct frame_info *next_frame = NULL;
  struct value *val = NULL;
  ULONGEST to = (ULONGEST)0ULL;

  /* Special registers are frame-independent by construction. But they must be
     evaluated lazily so that we get both value and address if needed. */
  if (regnum == cuda_special_regnum (gdbarch))
    {
      val = default_value_from_register (type, regnum, frame);
      set_value_lazy (val, 1);
      return val;
    }

  /* That should not happen. But the DWARF info may encode locations with
     DW_OP_regx, which means that the variable is in a register, no matter the
     call stack. This is not true. The register will be caller/callee-saved and
     its value (and address!) will be retrieved from the stack. We correct the
     situation here. 'Val' will be a lval_memory, not a lval_register in this
     case. */
  if (level > 0)
    {
      next_frame = get_next_frame (frame);
      val = cuda_frame_prev_register (next_frame, NULL, regnum);
      if (type && !TYPE_CUDA_REG(type))
        deprecated_set_value_type (val, type);
      return val;
    }

  val = default_value_from_register (type, regnum, frame);
  return val;
}

static CORE_ADDR
cuda_unwind_pc (struct gdbarch *gdbarch, struct frame_info *next_frame)
{
  CORE_ADDR pc;

  pc = cuda_frame_prev_pc (next_frame);

  if (frame_debug >= 3)
    fprintf_unfiltered (gdb_stdlog,
                        "{ cuda_unwind_pc (next_frame=n/a) -> %s }\n",
                        hex_string (pc));

  return pc;
}

static const gdb_byte *
cuda_breakpoint_from_pc (struct gdbarch *gdbarch, CORE_ADDR *pc, int *len)
{
  return NULL;
}

static struct gdbarch *
cuda_gdbarch_init (struct gdbarch_info info, struct gdbarch_list *arches)
{
  struct gdbarch      *gdbarch;
  struct gdbarch_tdep *tdep;

  /* If there is already a candidate, use it.  */
  arches = gdbarch_list_lookup_by_info (arches, &info);
  if (arches != NULL)
    return arches->gdbarch;

  /* Allocate space for the new architecture.  */
  tdep = XCALLOC (1, struct gdbarch_tdep);
  gdbarch = gdbarch_alloc (&info, tdep);

  /* Set extra CUDA architecture specific information */
  tdep->num_regs  = 128 + 1;
  tdep->sp_regnum       = 1;   /* ABI only, SP is in R1 */
  tdep->first_rv_regnum = 4;   /* ABI only, First RV is in R4, also used to pass args */
  tdep->last_rv_regnum  = 15;  /* ABI only, Last RV is in R15, also used to pass args */
  tdep->rz_regnum       = 63;  /* ABI only, Zero is in R63 */
  tdep->pc_regnum       = 128; /* PC is after the last user register */

  tdep->num_pseudo_regs    = 3;
  tdep->special_regnum     = 129;
  tdep->invalid_lo_regnum  = 130;
  tdep->invalid_hi_regnum  = 131;

  tdep->max_reg_rv_size = (tdep->last_rv_regnum - tdep->first_rv_regnum + 1) * 4;
  tdep->ptr_size = TARGET_CHAR_BIT * sizeof (CORE_ADDR); /* 32 or 64 bits */

  /* Data types.  */
  set_gdbarch_char_signed (gdbarch, 0);
  set_gdbarch_ptr_bit (gdbarch, tdep->ptr_size);
  set_gdbarch_addr_bit (gdbarch, 64);
  set_gdbarch_short_bit (gdbarch, 16);
  set_gdbarch_int_bit (gdbarch, 32);
  set_gdbarch_long_bit (gdbarch, 64);
  set_gdbarch_long_long_bit (gdbarch, 64);
  set_gdbarch_float_bit (gdbarch, 32);
  set_gdbarch_double_bit (gdbarch, 64);
  set_gdbarch_long_double_bit (gdbarch, 128);
  set_gdbarch_float_format (gdbarch, floatformats_ieee_single);
  set_gdbarch_double_format (gdbarch, floatformats_ieee_double);
  set_gdbarch_long_double_format (gdbarch, floatformats_ieee_double);

  /* Registers and Memory */
  set_gdbarch_num_regs        (gdbarch, tdep->num_regs);
  set_gdbarch_num_pseudo_regs (gdbarch, tdep->num_pseudo_regs);

  set_gdbarch_pc_regnum  (gdbarch, tdep->pc_regnum);
  set_gdbarch_ps_regnum  (gdbarch, -1);
  set_gdbarch_sp_regnum  (gdbarch, -1);
  set_gdbarch_fp0_regnum (gdbarch, -1);

  set_gdbarch_dwarf2_reg_to_regnum (gdbarch, cuda_reg_to_regnum);

  set_gdbarch_pseudo_register_write (gdbarch, cuda_pseudo_register_write);
  set_gdbarch_pseudo_register_read  (gdbarch, cuda_pseudo_register_read);

  set_gdbarch_read_pc  (gdbarch, NULL);
  set_gdbarch_write_pc (gdbarch, NULL);

  set_gdbarch_register_name (gdbarch, cuda_register_name);
  set_gdbarch_register_type (gdbarch, cuda_register_type);

  set_gdbarch_print_registers_info (gdbarch, cuda_print_registers_info);
  set_gdbarch_print_float_info     (gdbarch, NULL);
  set_gdbarch_print_vector_info    (gdbarch, NULL);

  set_gdbarch_convert_register_p  (gdbarch, cuda_convert_register_p);
  set_gdbarch_register_to_value   (gdbarch, cuda_register_to_value);
  set_gdbarch_value_to_register   (gdbarch, cuda_value_to_register);
  set_gdbarch_value_from_register (gdbarch, cuda_value_from_register);

  /* Pointers and Addresses */
  set_gdbarch_fetch_pointer_argument (gdbarch, NULL);

  /* Address Classes */
  set_gdbarch_address_class_name_to_type_flags(gdbarch,
                                               cuda_address_class_name_to_type_flags);
  set_gdbarch_address_class_type_flags_to_name(gdbarch,
                                               cuda_address_class_type_flags_to_name);
  set_gdbarch_address_class_type_flags (gdbarch,
                                        cuda_address_class_type_flags);

  /* Register Representation */
  /* Frame Interpretation */
  set_gdbarch_skip_prologue (gdbarch, cuda_skip_prologue);
  set_gdbarch_inner_than (gdbarch, core_addr_lessthan);
  set_gdbarch_frame_align (gdbarch, NULL);
  set_gdbarch_frame_red_zone_size (gdbarch, 0);
  set_gdbarch_frame_args_skip (gdbarch, 0);
  set_gdbarch_unwind_pc (gdbarch, cuda_unwind_pc);
  set_gdbarch_unwind_sp (gdbarch, NULL);
  set_gdbarch_frame_num_args (gdbarch, NULL);
  set_gdbarch_return_value (gdbarch, cuda_abi_return_value);
  frame_unwind_append_unwinder (gdbarch, &cuda_frame_unwind);
  frame_base_append_sniffer (gdbarch, cuda_frame_base_sniffer);
  frame_base_set_default (gdbarch, &cuda_frame_base);
  dwarf2_append_unwinders (gdbarch);
  dwarf2_frame_set_adjust_regnum (gdbarch, cuda_adjust_regnum);

  /* Inferior Call Setup */
  set_gdbarch_dummy_id (gdbarch, NULL);
  set_gdbarch_push_dummy_call (gdbarch, NULL);

  set_gdbarch_regset_from_core_section (gdbarch, NULL);
  set_gdbarch_skip_permanent_breakpoint (gdbarch, NULL);
  set_gdbarch_fast_tracepoint_valid_at (gdbarch, NULL);
  set_gdbarch_decr_pc_after_break (gdbarch, 0);
  set_gdbarch_max_insn_length (gdbarch, 8);

  /* Instructions */
  set_gdbarch_print_insn (gdbarch, cuda_print_insn);
  set_gdbarch_relocate_instruction (gdbarch, NULL);
  set_gdbarch_breakpoint_from_pc   (gdbarch, cuda_breakpoint_from_pc);

  /* CUDA - no address space management */
  set_gdbarch_has_global_breakpoints (gdbarch, 1);

  // We hijack the linux siginfo type for the CUDA target on both Mac & Linux
  set_gdbarch_get_siginfo_type (gdbarch, linux_get_siginfo_type);

  return gdbarch;
}

struct gdbarch *
cuda_get_gdbarch (void)
{
  struct gdbarch_info info;

  if (!cuda_gdbarch)
    {
      gdbarch_info_init (&info);
      info.bfd_arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
      cuda_gdbarch = gdbarch_find_by_info (info);
    }

  return cuda_gdbarch;
}

void
_initialize_cuda_tdep (void)
{
  register_gdbarch_init (bfd_arch_m68k, cuda_gdbarch_init);
}

bool
cuda_is_cuda_gdbarch (struct gdbarch *arch)
{
  if (gdbarch_bfd_arch_info (arch)->arch == bfd_arch_m68k)
    return true;
  else
    return false;
}

/********* Session Management **********/

int
cuda_gdb_session_create (void)
{
  int ret = 0;
  bool override_umask = false;
  bool dir_exists = false;

  /* Check if the previous session was cleaned up */
  if (cuda_gdb_session_dir[0] != '\0')
    error (_("The directory for the previous CUDA session was not cleaned up. "
             "Try deleting %s and retrying."), cuda_gdb_session_dir);

  cuda_gdb_session_id++;

  snprintf (cuda_gdb_session_dir, CUDA_GDB_TMP_BUF_SIZE,
            "%s/session%d", cuda_gdb_tmpdir_getdir (),
            cuda_gdb_session_id);

  cuda_trace ("new session %d created", cuda_gdb_session_id);

  ret = cuda_gdb_dir_create (cuda_gdb_session_dir, S_IRWXU | S_IRWXG,
                             override_umask, &dir_exists);

  if (!ret && dir_exists)
    error (_("A stale CUDA session directory was found. "
             "Try deleting %s and retrying."), cuda_gdb_session_dir);

  return ret;
}

void
cuda_gdb_session_destroy (void)
{
  cuda_gdb_dir_cleanup_files (cuda_gdb_session_dir);

  rmdir (cuda_gdb_session_dir);

  memset (cuda_gdb_session_dir, 0, CUDA_GDB_TMP_BUF_SIZE);
}

uint32_t
cuda_gdb_session_get_id (void)
{
  return cuda_gdb_session_id;
}

const char *
cuda_gdb_session_get_dir (void)
{
    return cuda_gdb_session_dir;
}

