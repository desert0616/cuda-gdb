/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2013 NVIDIA Corporation
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

#include "defs.h"
#include "gdbtypes.h"
#include "gdbarch.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#ifdef __APPLE__
#include <libproc.h>
#include <IOKit/IOKitLib.h>

/* Maximum length of IOKit registry objects list */
#define MAX_OBJLIST_LEN 128

/* Darwin process name length */
#define DARWIN_PROC_NAME_LEN 128

/* Statuses of CUDA_GPU devices */
#define CUDA_GPU_PRESENT 1
#define CUDA_GPU_HAS_DISPLAY 2
#define CUDA_GPU_HAS_COMPUTE_USERS 4
#define CUDA_GPU_HAS_FB_USERS 8
#define CUDA_GPU_HAS_ALL 15

static IOReturn
DarwinGetObjects(mach_port_t port, int *count, io_object_t *objects, const char *name)
{
  IOReturn rc;
  CFMutableDictionaryRef dict = NULL;
  io_iterator_t iterator = 0;
  io_object_t object = 0;

  dict = IOServiceMatching (name);
  if (!dict)
    return kIOReturnNoMemory;
  rc = IOServiceGetMatchingServices (port, dict, &iterator);
  if (rc != kIOReturnSuccess)
    {
      CFRelease (dict);
      return rc;
    }
  *count = 0;
  while ((object = IOIteratorNext (iterator)))
    {
      if (objects)
        objects[*count] = object;
      if ((*count)++ == MAX_OBJLIST_LEN) break;
    }
  return IOObjectRelease (iterator);
}

static IOReturn
DarwinGetParentOfType(io_object_t object, io_object_t *pParent, const char *type)
{
  IOReturn rc;
  io_object_t parent;
  io_name_t class_name;

  assert (pParent != NULL);

  while (object)
    {
      rc = IOObjectGetClass (object, class_name);
      if (rc != kIOReturnSuccess) return rc;
      if (!type || strcmp (class_name, type) == 0)
        {
          *pParent = object;
          return kIOReturnSuccess;
        }
      rc = IORegistryEntryGetParentEntry (object, kIOServicePlane, &parent);
      if ( rc != kIOReturnSuccess) return rc;
      object = parent;
    }
  return kIOReturnNoDevice;
}

static IOReturn
DarwinGetObjectChildsOfType(io_object_t obj, int *count, io_object_t *objects, const char *type)
{
  IOReturn rc;
  io_iterator_t iterator;
  io_object_t child;
  io_name_t class_name;

  rc = IORegistryEntryGetChildIterator (obj, kIOServicePlane, &iterator);
  if (rc != kIOReturnSuccess)
    return rc;

  *count = 0;
  while ((child = IOIteratorNext (iterator)))
    {
      rc = IOObjectGetClass (child, class_name);
      if (rc != kIOReturnSuccess) continue;
      if (type && strcmp (class_name, type)!=0) continue;
      if (objects)
        objects[*count] = child;
      if ((*count)++ == MAX_OBJLIST_LEN) break;
    }
  return IOObjectRelease (iterator);
}

static int
DarwinGetChildsOfTypeCount(io_object_t obj, const char *type)
{
  IOReturn rc;
  int count;

  rc = DarwinGetObjectChildsOfType (obj, &count, NULL, type);
  return rc != kIOReturnSuccess ? -1 : count;
}


static IOReturn
DarwinGetObjectsPCIParent (io_object_t *list, int count)
{
  int i;
  io_object_t parent;
  IOReturn rc;

  for(i = 0; i < count; i++)
    {
      rc = DarwinGetParentOfType (list[i], &parent, "IOPCIDevice");
      if (rc != kIOReturnSuccess)
        return rc;
      if (parent == 0)
        return kIOReturnNoDevice;
      list[i] = parent;
    }
  return kIOReturnSuccess;
}

static int
object_index_in_list (io_object_t *list, io_object_t obj)
{
  int i;

  for (i = 0; list[i] != 0; i++)
    if (list[i] == obj) return i;
  return -1;
}


/*
 * Tries to determine if GPU used for graphics is also used for CUDA
 * If any of the system calls fails, it assumes that's the case.
 * GPU is considered busy with both compute and graphics
 * if following conditions are met:
 * - Display is attached to a given GPU
 * - At least one frame-buffer client is using this GPU (i.e. WindowServer is running)
 * - At least one compute client is using this GPU
 */
bool
cuda_darwin_compute_gpu_used_for_graphics(void)
{
 int count;
 int i, idx;
 io_object_t displays[MAX_OBJLIST_LEN+1];
 io_object_t cuda_gpus[MAX_OBJLIST_LEN+1];
 int cuda_gpu_status[MAX_OBJLIST_LEN+1];
 io_object_t nvdas[MAX_OBJLIST_LEN+1];
 io_object_t parent;
 IOReturn rc;

 /* Get NVKernel objects */
 memset (cuda_gpus, 0, sizeof(cuda_gpus));
 memset (cuda_gpu_status, 0, sizeof(cuda_gpu_status));
 rc = DarwinGetObjects (kIOMasterPortDefault, &count, cuda_gpus, "NVKernel");
 if (rc != kIOReturnSuccess || count == 0)
    return true;
  rc = DarwinGetObjectsPCIParent (cuda_gpus, count);
  if (rc != kIOReturnSuccess)
    return true;
  for(i = 0; i < count; i++)
    cuda_gpu_status[i] = CUDA_GPU_PRESENT;

  /* Get IODisplayConnect objects */
  memset (displays, 0, sizeof(displays));
  rc = DarwinGetObjects (kIOMasterPortDefault, &count, displays, "IODisplayConnect");
  if (rc != kIOReturnSuccess)
    return true;

  /* Not a single display attached*/
  if (count == 0) return false;
  rc = DarwinGetObjectsPCIParent (displays, count);
  if (rc != kIOReturnSuccess)
    return true;

  /* Map displays to GPUs*/
  for (i = 0;i < count; i++)
    {
      idx = object_index_in_list (cuda_gpus, displays[i]);
      /* Display attached to non-nvidia GPU*/
      if (idx < 0) continue;

      cuda_gpu_status[idx] |= CUDA_GPU_HAS_DISPLAY;
    }

  /* Get NVDA objects */
  memset (nvdas, 0, sizeof(nvdas));
  rc = DarwinGetObjects (kIOMasterPortDefault, &count, nvdas, "NVDA");
  if (rc != kIOReturnSuccess || count == 0)
    return true;

  /* Map framebuffer and compute users to GPUs*/
  for (i=0;i<count;i++)
    {
      rc = DarwinGetParentOfType (nvdas[i], &parent, "IOPCIDevice");
      if (rc != kIOReturnSuccess || parent == 0)
        return true;
      idx = object_index_in_list (cuda_gpus, parent);
      if (idx < 0)
        return true;
      rc = DarwinGetChildsOfTypeCount (nvdas[i], "NVDAUser");
      if (rc < 0) return true;
      if (rc > 0) cuda_gpu_status[idx] |= CUDA_GPU_HAS_COMPUTE_USERS;
      if (rc < 0) return true;
      if (rc > 0) cuda_gpu_status[idx] |= CUDA_GPU_HAS_FB_USERS;
    }

  /* Check if there are any GPUs that has compute and framebuffer clients as well as display attached */
  for (idx = 0; (cuda_gpu_status[idx]&CUDA_GPU_PRESENT) != 0; idx++)
    if (cuda_gpu_status[idx] == CUDA_GPU_HAS_ALL) return true;
	
  return false;
}

static int 
darwin_find_session_leader (int *pPid, char *name)
{
  int pid, rc;
  struct proc_bsdinfo bsdinfo;

  pid = getpid();
  while (pid>0)
    {
      rc = proc_pidinfo (pid, PROC_PIDTBSDINFO, 0, &bsdinfo, sizeof(bsdinfo));
      if (rc != sizeof(bsdinfo)) return -1;
      if ((bsdinfo.pbi_flags & PROC_FLAG_SLEADER) == PROC_FLAG_SLEADER) 
        {
          *pPid = pid;
          rc = proc_name (pid, name, DARWIN_PROC_NAME_LEN);
          if (rc < 0) return rc;
          return 0;
        }
        pid = bsdinfo.pbi_ppid;
    }
  return -1;
}

bool
cuda_darwin_is_launched_from_ssh_session(void)
{
  int rc,pid;
  char name[DARWIN_PROC_NAME_LEN];

  rc =  darwin_find_session_leader (&pid, name);

  /* If application is launched from Terminal, its session leader is /sbin/login, which has suid bit*/
  if (rc != 0) return false;

  /* Session leader for anything launched from bash or ourselves*/
  return (strcmp (name, "bash")==0 || pid == getpid()) ? true : false;
}

#endif /*__APPLE__*/
