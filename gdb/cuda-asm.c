/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2017 NVIDIA Corporation
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

#include <string.h>
#include <ctype.h>

#include "defs.h"
#include "frame.h"
#include "common-defs.h"
#include "exceptions.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-modules.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-options.h"

/******************************************************************************
 *
 *                        One PC-Instruction Mapping
 *
 *****************************************************************************/

typedef struct inst_st      *inst_t;

#define INSN_MAX_LENGTH 80

struct inst_st {
  uint64_t      pc;      /* the PC of the disassembled instruction */
  char         *text;    /* the dissassembled instruction */
  uint32_t      size;    /* size of the instruction in bytes */
  inst_t next;           /* the next element in the list */
};

static inst_t
inst_create (uint64_t pc, const char *text, uint32_t size, inst_t next)
{
  inst_t inst;
  int len;

  gdb_assert (text);

  len = strlen (text);
  inst = (inst_t) xmalloc (sizeof *inst);
  inst->text = (char *) xmalloc (len + 1);

  inst->pc   = pc;
  inst->text = strncpy (inst->text, text, len + 1);
  inst->next = next;
  inst->size = size;

  return inst;
}

static void
inst_destroy (inst_t inst)
{
  xfree (inst->text);
  xfree (inst);
}


/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

struct disasm_cache_st {
  bool       cached;            /* have we already tried to populate this disasm_cache */
  uint64_t   entry_pc;          /* entry PC of the function being cached */
  inst_t     head;              /* head of the list of instructions */
};

disasm_cache_t
disasm_cache_create (void)
{
  disasm_cache_t disasm_cache;

  disasm_cache = (disasm_cache_t) xmalloc (sizeof *disasm_cache);

  disasm_cache->cached     = false;
  disasm_cache->entry_pc   = 0ULL;
  disasm_cache->head       = NULL;

  return disasm_cache;
}

void
disasm_cache_flush (disasm_cache_t disasm_cache)
{
  inst_t inst = disasm_cache->head;
  inst_t next_inst;

  while (inst)
    {
      next_inst = inst->next;
      inst_destroy (inst);
      inst = next_inst;
    }

  disasm_cache->cached     = false;
  disasm_cache->entry_pc   = 0ULL;
  disasm_cache->head       = NULL;
}

void
disasm_cache_destroy (disasm_cache_t disasm_cache)
{
  disasm_cache_flush (disasm_cache);

  xfree (disasm_cache);
}

extern char *gdb_program_name;

static int
gdb_program_dir_len (void)
{
  static int len = -1;
  int cnt;

  if (len >=0)
    return len;

  len = strlen(gdb_program_name);
  for (cnt=len; cnt > 1 && gdb_program_name[cnt-1] != '/'; cnt--);
  if (cnt > 1)
    len = cnt;

  return len;
}

/**
 * Search for executable in PATH, cuda-gdb launch folder or current folder
 */
static bool exists(const char *fname)
{
  struct stat buf;
  return stat (fname, &buf) == 0;
}

static const char *find_executable(const char *name)
{
  static char return_path[1024];
  char path[4096];
  char *dir;

  /* Save PATH to local variable because strtok() alters the string */
  strncpy (path, getenv("PATH"), sizeof(path));
  path[sizeof(path)-1] = 0;

  for(dir = strtok (path, ":"); dir; dir = strtok (NULL, ":"))
    {
      snprintf (return_path, sizeof (return_path), "%s/%s", dir, name);
      if (exists (return_path))
        return return_path;
    }

  snprintf (return_path, sizeof (return_path), "%.*s%s",
            gdb_program_dir_len(), gdb_program_name, name);
  if (exists (return_path))
    return return_path;

  return name;
}

static char *find_cuobjdump (void)
{
  static char cuobjdump_path[1024];
  static bool cuobjdump_path_initialized = false;

  if (cuobjdump_path_initialized)
    return cuobjdump_path;

  strncpy (cuobjdump_path, find_executable ("cuobjdump"), sizeof (cuobjdump_path));
  cuobjdump_path_initialized = true;
  return cuobjdump_path;
}

static bool
disasm_cache_parse_line(const char *line, char *insn, uint64_t *offs)
{
  #define INSN_HEX_SIGNATURE "/* 0x"
  #define INSN_HEX_LENGTH strlen(INSN_HEX_SIGNATURE)
  char *size_start_ptr = strstr(line, INSN_HEX_SIGNATURE);
  char *size_end_ptr = size_start_ptr ? strchr (size_start_ptr + INSN_HEX_LENGTH, ' ') : NULL;
  char *offs_start_ptr = strstr(line, "/*");
  char *offs_end_ptr = offs_start_ptr + 2;
  char *semi_colon;
  unsigned long length;

  memset (insn, 0, INSN_MAX_LENGTH);
  *offs = (uint64_t)-1LL;

  /* If instruction size signature can not be found, return false*/
  if (!size_end_ptr)
    return false;

  /* Check that there is nothing but spaces before the instruction offset location */
  while (isspace (*line)) ++line;
  if (line != offs_start_ptr)
    return false;

  /* If offs_ptr is NULL - return empty line instruction */
  if (!offs_start_ptr || offs_start_ptr == size_start_ptr)
      return true;

  *offs = strtoull (offs_start_ptr + 2, &offs_end_ptr, 16);
  /* Fail If field containing offset is present, but has invalid hexadecimal number */
  if (offs_end_ptr == offs_start_ptr + 2)
    return false;
  if (offs_end_ptr[0] != '*' || offs_end_ptr[1] != '/')
    return false;

  /* Strip whitespaces before the start of the assembly mnemonic */
  for (offs_end_ptr += 2; isspace (*offs_end_ptr); ++offs_end_ptr);

  /* Ignore everything after semicolon (if present)*/
  semi_colon = strchr (offs_end_ptr, ';');
  if (semi_colon)
    size_start_ptr = semi_colon;
  else
    /* Strip whitespaces and the end of the assembly mnemonic */
    for (--size_start_ptr; isspace (*size_start_ptr); --size_start_ptr);

  length = (unsigned long)size_start_ptr - (unsigned long)offs_end_ptr;

  /* If necessary, trim mnemonic length*/
  if (length >= INSN_MAX_LENGTH) length = INSN_MAX_LENGTH-1;
  memcpy (insn, offs_end_ptr, length);

  return true;
}

static uint32_t
disasm_get_inst_size (uint64_t pc)
{
    uint32_t inst_size = 0;
    uint32_t devId = cuda_current_device ();

    cuda_api_disassemble (devId, pc, &inst_size, NULL, 0);
    return inst_size;
}

static void
disasm_cache_populate_from_elf_image (disasm_cache_t disasm_cache, uint64_t pc)
{
  kernel_t kernel;
  module_t module;
  elf_image_t elf_img;
  struct objfile *objfile;
  uint32_t devId, inst_size;
  uint64_t ofst = 0, entry_pc = 0;
  FILE *sass;
  char command[1024], line[1024], header[1024];
  char *filename;
  const char *function_name;
  char *function_base_name;
  char text[INSN_MAX_LENGTH];
  bool header_found = false;

  entry_pc = get_pc_function_start (pc);
  /* Exit early if PC does not belong to the code segment */
  if (entry_pc == 0)
    return;
  /* early exit if already cached */
  if (disasm_cache->cached && disasm_cache->entry_pc == entry_pc)
    return;

  devId = cuda_current_device ();
  inst_size = device_get_inst_size (devId);
  if (!inst_size)
    {
      inst_size = disasm_get_inst_size (entry_pc);
      if (!inst_size)
        throw_error (GENERIC_ERROR, "Cannot find the instruction size while disassembling.");
      device_set_inst_size (devId, inst_size);
    }

  disasm_cache_flush (disasm_cache);
  disasm_cache->cached = true;
  disasm_cache->entry_pc = entry_pc;

  /* collect all the necessary data */
  kernel      = cuda_current_kernel ();
  module      = kernel_get_module (kernel);
  elf_img     = module_get_elf_image (module);
  objfile     = cuda_elf_image_get_objfile (elf_img);
  filename    = objfile->original_name;
  function_name = cuda_find_function_name_from_pc (pc, false);

  /* Could not disasemble outside of the symbol boundaries */
  if (!function_name)
    return;

  /* generate the dissassembled code using cuobjdump if available */
  snprintf (command, sizeof (command), "%s --function '%s' --dump-sass '%s'",
            find_cuobjdump(), function_name, filename);
  sass = popen (command, "r");

  if (!sass)
    throw_error (GENERIC_ERROR, "Cannot disassemble from the ELF image.");

  /* discard the function arguments if specified */
  function_base_name = strdup (function_name);
  function_base_name = strtok (function_base_name, "(");

  /* find the wanted function in the cuobjdump output */
  snprintf (header, sizeof (header), "\t\tFunction : %s\n", function_base_name);
  while (!header_found && fgets (line, sizeof (line), sass) != NULL)
    if (strncmp (line, header, strlen (header)) == 0)
      header_found = true;
  xfree (function_base_name);

  /* return if failed to detect the function header */
  if (!header_found)
    {
      pclose (sass);
      throw_error (GENERIC_ERROR, "Cannot find the function header while disassembling.");
    }

  /* parse the sass output and insert each instruction individually */
  while (fgets (line, sizeof (line), sass) != NULL)
    {
      /* stop reading at the first white line */
      if (strcmp (line, "\n") == 0)
        break;

       if (!disasm_cache_parse_line (line, text, &ofst))
         continue;
       if (ofst == (uint64_t)-1LL)
           ofst = disasm_cache->head ? disasm_cache->head->pc + disasm_cache->head->size : entry_pc;
       else
           ofst += entry_pc;

      /* add the instruction to the cache at the found offset */
      disasm_cache->head = inst_create (ofst, text, inst_size, disasm_cache->head);
    }

  /* close the sass file */
  pclose (sass);

  /* we expect to always being able to diassemble at least one instruction */
  if (cuda_options_debug_strict () && !disasm_cache->head)
    throw_error (GENERIC_ERROR, "Unable to disassemble a single device instruction.");
}

static void
disasm_cache_read_from_device_memory (disasm_cache_t disasm_cache, uint64_t pc)
{
  char buf[512];
  uint32_t devId;
  uint32_t inst_size;

  /* no caching */
  disasm_cache_flush (disasm_cache);

  if (!cuda_initialized)
    return;

  buf[0] = 0;
  devId = cuda_current_device ();
  cuda_api_disassemble (devId, pc, &inst_size, buf, sizeof (buf));

  disasm_cache->head = inst_create (pc, buf, inst_size, disasm_cache->head);
}

const char *
disasm_cache_find_instruction (disasm_cache_t disasm_cache,
                               uint64_t pc, uint32_t *inst_size)
{
  inst_t inst = NULL;

  if (!cuda_focus_is_device ())
    return NULL;

  /* compute the disassembled instruction */
  if (cuda_options_disassemble_from_elf_image ())
    disasm_cache_populate_from_elf_image (disasm_cache, pc);
  else
    disasm_cache_read_from_device_memory (disasm_cache, pc);


  /* find the cached disassembled instruction. */
  for (inst = disasm_cache->head; inst; inst = inst->next)
    if (inst->pc == pc)
      break;

  /* return the instruction or NULL if not found */
  if (inst && inst->text)
    {
      *inst_size = inst->size;
      return inst->text;
    }
  else
    {
      *inst_size = 4;
      return NULL;
    }
}
