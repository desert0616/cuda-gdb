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

#ifndef _CUDA_API_H
#define _CUDA_API_H 1

#include "cudadebugger.h"

/* Initialization */
int  cuda_api_get_api (void);
int  cuda_api_initialize (void);
void cuda_api_finalize (void);

/* Device Execution Control */
void cuda_api_suspend_device (uint32_t dev);
void cuda_api_resume_device (uint32_t dev);
void cuda_api_single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *warp_mask);

/* Breakpoints */
bool cuda_api_set_breakpoint (uint32_t dev, uint64_t addr);
bool cuda_api_unset_breakpoint (uint32_t dev, uint64_t addr);

/* Device State Inspection */
void cuda_api_read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *grid_id);
void cuda_api_read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
void cuda_api_read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);
void cuda_api_read_broken_warps (uint32_t dev, uint32_t sm, uint64_t *brokenWarpsMask);
void cuda_api_read_valid_warps (uint32_t dev, uint32_t sm, uint64_t *valid_warps);
void cuda_api_read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes);
void cuda_api_read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes);
void cuda_api_read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_global_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
bool cuda_api_read_pinned_memory (uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_texture_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t id, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz);
void cuda_api_read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
void cuda_api_read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
void cuda_api_read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
void cuda_api_read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);
void cuda_api_read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
void cuda_api_read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
void cuda_api_read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t level, uint64_t *ra);

/* Device State Alteration */
void cuda_api_write_global_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
bool cuda_api_write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);

/* Grid Properties */
void cuda_api_get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim);
void cuda_api_get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim);
void cuda_api_get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
void cuda_api_get_elf_image (uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint64_t *size);
void cuda_api_get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking);

/* Device Properties */
void cuda_api_get_device_type (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_sm_type (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_num_devices (uint32_t *numDev);
void cuda_api_get_num_sms (uint32_t dev, uint32_t *numSMs);
void cuda_api_get_num_warps (uint32_t dev, uint32_t *numWarps);
void cuda_api_get_num_lanes (uint32_t dev, uint32_t *numLanes);
void cuda_api_get_num_registers (uint32_t dev, uint32_t *numRegs);

/* DWARF-related routines */
void cuda_api_disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize);
void cuda_api_is_device_code_address (uint64_t addr, bool *is_device_address);
bool cuda_api_lookup_device_code_symbol (char *name, uint64_t *addr);

/* Events */
void cuda_api_set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback);
void cuda_api_acknowledge_events (void);
void cuda_api_get_next_event (CUDBGEvent *event);

#endif

