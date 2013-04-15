/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2011 NVIDIA Corporation
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

#ifndef _CUDA_TDEP_H
#define _CUDA_TDEP_H 1

#include "bfd.h"
#include "elf-bfd.h"
#include "defs.h"
#include "cudadebugger.h"
#include "gdbarch.h"
#include "dis-asm.h"
#include "environ.h"
#include "cuda-api.h"
#include "cuda-coords.h"
#include "cuda-defs.h"
#include "cuda-kernel.h"
#include "cuda-modules.h"
#include "progspace.h"

extern bool cuda_elf_path; /* REMOVE THIS ONCE CUDA ELF PATH IS COMPLETE! */

/* CUDA - skip prologue
   REMOVE ONCE TRANSITION TESLA KERNELS HAVE PROLOGUES ALL THE TIME */
extern bool cuda_producer_is_open64;

#define DEFAULT_PROMPT   "(cuda-gdb) "
#define GDBINIT_FILENAME ".cuda-gdbinit"

/*---------------------------- CUDA ELF Specification --------------------------*/

#define EV_CURRENT                   1
#define ELFOSABI_CUDA             0x33
#define CUDA_ELFOSABIV_16BIT         0  /* 16-bit ctaid.x size */
#define CUDA_ELFOSABIV_32BIT         1  /* 32-bit ctaid.x size */
#define CUDA_ELFOSABIV_RELOC         2  /* ELFOSABIV_32BIT + All relocators in DWARF */
#define CUDA_ELFOSABIV_ABI           3  /* ELFOSABIV_RELOC + Calling Convention */
#define CUDA_ELFOSABIV_SYSCALL       4  /* ELFOSABIV_ABI + Improved syscall relocation */
#define CUDA_ELFOSABIV_LATEST        4  /* Latest ABI version*/
#define CUDA_ELF_TEXT_PREFIX  ".text."  /* CUDA ELF text section format: ".text.KERNEL" */

/*Return values that exceed 384-bits in size are returned in memory.
   (R4-R15 = 12 4-byte registers = 48-bytes = 384-bits that can be
   used to return values in registers). */
#define CUDA_ABI_MAX_REG_RV_SIZE  48 /* Size in bytes */

/*------------------------------ Type Declarations -----------------------------*/

typedef struct {
  bool valid;
  uint32_t value;
  bool recoverable;
} cuda_exception_t;

extern cuda_exception_t cuda_exception;

typedef enum {
  cuda_bp_none = 0,
  cuda_bp_runtime_api, /* Transition from host stub code to device code */
  cuda_bp_driver_api,  /* Always dynamically resolved (initially pending) */
} cuda_bptype_t;

#define CUDA_MAX_NUM_RESIDENT_BLOCKS_PER_GRID 256
#define CUDA_MAX_NUM_RESIDENT_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_RESIDENT_THREADS (CUDA_MAX_NUM_RESIDENT_BLOCKS_PER_GRID * CUDA_MAX_NUM_RESIDENT_THREADS_PER_BLOCK)

struct cuda_frame_info
{
  /* Is it a cuda internal frame? */
  bool cuda_internal_p;
  bool cuda_internal;

  /* Is it a cuda global kernel (entrypoint) frame? */
  bool cuda_device_p;
  bool cuda_device;

  /* Is it a cuda runtime entry point frame? */
  bool cuda_runtime_entrypoint_p;
  bool cuda_runtime_entrypoint;

  /* Is it a cuda device syscall frame? */
  bool cuda_device_syscall_p;
  bool cuda_device_syscall;

  /* Frame level when hiding cuda runtime frames */
  bool cuda_level_p;
  int  cuda_level;
};

typedef enum return_value_convention rvc_t;

extern CuDim3 gridDim;
extern CuDim3 blockDim;

typedef bool (*cuda_thread_func)(cuda_coords_t *, void *);

/*------------------------------ Global Variables ------------------------------*/

extern bool cuda_debugging_enabled;
struct gdbarch * cuda_get_gdbarch (void);
bool cuda_is_cuda_gdbarch (struct gdbarch *);

cuda_coords_t cuda_coords_current;

/* Offsets of the CUDA built-in variables */
#define CUDBG_BUILTINS_BASE                        ((CORE_ADDR) 0)
#define CUDBG_THREADIDX_OFFSET           (CUDBG_BUILTINS_BASE - 6)
#define CUDBG_BLOCKIDX_OFFSET         (CUDBG_THREADIDX_OFFSET - 6)
#define CUDBG_BLOCKDIM_OFFSET          (CUDBG_BLOCKIDX_OFFSET - 6)
#define CUDBG_GRIDDIM_OFFSET           (CUDBG_BLOCKDIM_OFFSET - 6)
#define CUDBG_WARPSIZE_OFFSET          (CUDBG_GRIDDIM_OFFSET - 32)
#define CUDBG_BUILTINS_MAX                 (CUDBG_WARPSIZE_OFFSET)

/*----------- Prototypes to avoid implicit declarations (hack-hack) ------------*/

extern bool cuda_initialized;

struct partial_symtab;
void switch_to_cuda_thread (cuda_coords_t *coords);
int  cuda_thread_select (char *, int);
void cuda_update_cudart_symbols (void);
void cuda_cleanup_cudart_symbols (void);
void cuda_update_convenience_variables (void);
void cuda_set_environment (struct gdb_environ *);

/*-------------------------------- Prototypes ----------------------------------*/

int  cuda_startup (void);
void cuda_kill (void);
void cuda_cleanup (void);
void cuda_final_cleanup (void *unused);
void cuda_initialize_target (void);
bool cuda_inferior_in_debug_mode (void);
void cuda_load_device_info (char *, struct partial_symtab *);

char *   cuda_find_kernel_name_from_pc (CORE_ADDR pc, bool demangle);
bool     cuda_breakpoint_hit_p (cuda_clock_t clock);
bool     cuda_exception_hit_p (cuda_exception_t *exception);
const char * cuda_exception_type_to_name (CUDBGException_t exception_type);

/*Frame Management */
const struct frame_unwind * cuda_frame_sniffer (struct frame_info *next_frame);
const struct frame_base * cuda_frame_base_sniffer (struct frame_info *next_frame);
bool cuda_frame_p (struct frame_info *next_frame);
bool cuda_frame_outermost_p (struct frame_info *next_frame);
int  cuda_frame_relative_level (struct frame_info *frame);
int  cuda_frame_is_internal (struct frame_info *fi);
int  cuda_frame_is_device (struct frame_info *fi);
int  cuda_frame_is_runtime_entrypoint (struct frame_info *fi);
int  cuda_frame_is_device_syscall (struct frame_info *fi);

/*Debugging */
void cuda_trace (char *fmt, ...);

/*----------------------------------------------------------------------------*/

/*Single-Stepping */
bool   cuda_sstep_is_active (void);
ptid_t cuda_sstep_ptid (void);
void   cuda_sstep_set_ptid (ptid_t ptid);
void   cuda_sstep_initialize (bool stepping);
void   cuda_sstep_execute (ptid_t ptid);
void   cuda_sstep_reset (bool sstep);
bool   cuda_sstep_kernel_has_terminated (void);

/*Registers */
bool          cuda_get_dwarf_register_string (reg_t reg, char *deviceReg, size_t sz);
void          cuda_regnum_pc_pre_hack (struct frame_info *fi);
void          cuda_regnum_pc_post_hack (void);
struct value *cuda_value_of_builtin_frame_phys_pc_reg (struct frame_info *frame);

/*Storage addresses and names */
void        cuda_print_lmem_address_type (void);
int         cuda_address_class_type_flags (int byte_size, int addr_class);

/*ABI/BFD/ELF/DWARF/objfile calls */
int             cuda_inferior_word_size ();
bool            cuda_is_bfd_cuda (bfd *obfd);
bool            cuda_is_bfd_version_call_abi (bfd *obfd);
bool            cuda_get_bfd_abi_version (bfd *obfd, unsigned int *abi_version);
bool            cuda_current_active_elf_image_uses_abi (void);
void            cuda_update_elf_images (void);
CORE_ADDR       cuda_dwarf2_func_baseaddr (struct objfile *objfile, char *func_name);
bool            cuda_find_function_pc_from_objfile (struct objfile *objfile, char *func_name, CORE_ADDR *func_addr);
bool            cuda_find_func_text_vma_from_objfile (struct objfile *objfile, char *func_name, CORE_ADDR *vma);
uint64_t        cuda_find_context_id (CORE_ADDR addr);
context_t       cuda_find_context_by_addr (CORE_ADDR addr);
bool            cuda_is_device_code_address (CORE_ADDR addr);

/*Segmented memory reads/writes */
int cuda_read_memory_partial (CORE_ADDR address, gdb_byte *buf, int len, struct type *type);
void cuda_read_memory  (CORE_ADDR address, struct value *val, struct type *type, int len);
int cuda_write_memory_partial (CORE_ADDR address, const gdb_byte *buf, struct type *type);
void cuda_write_memory (CORE_ADDR address, const gdb_byte *buf, struct type *type);

/*Breakpoints */
void cuda_resolve_breakpoints (elf_image_t elf_image);
void cuda_unresolve_breakpoints (uint64_t context);
int cuda_breakpoint_address_match (struct gdbarch *gdbarch,
                                   struct address_space *aspace1, CORE_ADDR addr1,
                                   struct address_space *aspace2, CORE_ADDR addr2);

/* Linux vs. Mac OS X */
bool cuda_platform_supports_tid (void);
int  cuda_gdb_get_tid (ptid_t ptid);
int  cuda_get_signo (void);
void cuda_set_signo (int signo);


#endif

