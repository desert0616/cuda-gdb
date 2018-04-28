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

/*Warning: this isn't intended as a standalone compile module! */

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#ifndef __ANDROID__
#include <sys/signal.h>
#else
#include <signal.h>
#endif
#include <defs.h>
#include <time.h>
#include <objfiles.h>
#include "block.h"
#include "gdbthread.h"
#include "language.h"
#include "demangle.h"
#include "regcache.h"
#include "arch-utils.h"
#include "buildsym.h"
#include "cuda-commands.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-parser.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "cuda-packet-manager.h"
#include "cuda-convvars.h"
#include "valprint.h"
#include "command.h"
#include "gdbcmd.h"
#include "observer.h"
#if defined(__linux__) && defined(GDB_NM_FILE)
#include "linux-nat.h"
#endif
#include "inf-child.h"
#include "cuda-linux-nat.h"
#include "top.h"
#include "event-top.h"

#define CUDA_NUM_CUDART_FRAME_ENTRIES 3

static struct {
  struct objfile *objfile;
} cuda_cudart_symbols;

/* Copy of the original host set of target operations. When a CUDA target op
   does not apply because we are dealing with the host code/cpu, use those
   routines instead. */
static struct target_ops host_target_ops;

static void cuda_nat_detach (struct target_ops *ops, const char *args,
			     int from_tty);

/* The whole Linux siginfo structure is presented to the user, but, internally,
   only the si_signo matters. We do not save the siginfo object. Instead we
   save only the signo. Therefore any read/write to any other field of the
   siginfo object will have no effect or will return 0. */
static enum target_xfer_status
cuda_nat_xfer_siginfo (struct target_ops *ops, enum target_object object,
                       const char *annex, gdb_byte *readbuf,
                       const gdb_byte *writebuf, ULONGEST offset, LONGEST len,
		       ULONGEST *xfered_len)
{
  /* the size of siginfo is not consistent between ptrace and other parts of
     GDB. On 32-bit Linux machines, the layout might be 64 bits. It does not
     matter for CUDA because only signo is used and the rest is set to zero. We
     just allocate 8 extra bytes and bypass the issue. On 64-bit Mac, the
     difference is 24 bytes. Therefore take the max of the 2 values. */
  gdb_byte buf[sizeof (siginfo_t) + 24];
  siginfo_t *siginfo = (siginfo_t *) buf;

  gdb_assert (object == TARGET_OBJECT_SIGNAL_INFO);
  gdb_assert (readbuf || writebuf);

  if (!cuda_focus_is_device ())
    return TARGET_XFER_E_IO;

  if (offset >= sizeof (buf))
    return TARGET_XFER_E_IO;

  if (offset + len > sizeof (buf))
    len = sizeof (buf) - offset;

  memset (buf, 0 , sizeof buf);

  if (readbuf)
    {
      siginfo->si_signo = cuda_get_signo ();
      memcpy (readbuf, siginfo + offset, len);
    }
  else
    {
      memcpy (siginfo + offset, writebuf, len);
      cuda_set_signo (siginfo->si_signo);
    }

  *xfered_len = len;
  return TARGET_XFER_OK;
}

static enum target_xfer_status
cuda_nat_xfer_partial (struct target_ops *ops,
                       enum target_object object, const char *annex,
                       gdb_byte *readbuf, const gdb_byte *writebuf,
                       ULONGEST offset, ULONGEST len, ULONGEST *xfered_len)
{
  enum target_xfer_status status = TARGET_XFER_E_IO;
  uint32_t dev, sm, wp, ln;
  *xfered_len = 0;

  /* Either readbuf or writebuf must be a valid pointer */
  gdb_assert (readbuf != NULL || writebuf != NULL);

  /* If focus is not set on device, call the host routines directly */
  if (!cuda_focus_is_device ())
    {
      status = host_target_ops.to_xfer_partial (ops, object, annex, readbuf,
                                                writebuf, offset, len,
						xfered_len);
#ifdef __arm__
      /*
       * FIXME - Temporary workaround for mmap()/ptrace() issue.
       * If xfer partial targets object other than memory and error is hit,
       * return right away to let cuda-gdb return the right error.
       */
       if (*xfered_len <= 0 && object != TARGET_OBJECT_MEMORY)
         return TARGET_XFER_OK;

       /*
       * If the host memory xfer operation fails (i.e. *xfered_len is 0),
       * fallthrough to see if the CUDA Debug API can access
       * the specified address.
       * This can happen with ordinary mmap'd allocations.
       */
      if (*xfered_len > 0)
	return status;
#else
      return status;
#endif
    }

  switch (object)
  {
    /* If focus is on GPU, still try to read the address using host routines,
       if it fails, see if this address is in pinned system memoryi, i.e. to
       system memory that was allocated by the inferior through the CUDA API */
    case TARGET_OBJECT_MEMORY:

      status = host_target_ops.to_xfer_partial (ops, object, annex, readbuf,
                                                writebuf, offset, len,
						xfered_len);
      if (*xfered_len)
        return TARGET_XFER_OK;

      if (readbuf ? cuda_api_read_pinned_memory  (offset, readbuf, len) :
		    cuda_api_write_pinned_memory (offset, writebuf, len))
	{
	  *xfered_len = len;
	  return TARGET_XFER_OK;
	}

      /* If all else failed, try to read memory from the device.  */
      if (cuda_coords_get_current_physical (&dev, &sm, &wp, &ln))
        return TARGET_XFER_E_IO;

      if (readbuf)
        cuda_api_read_local_memory (dev, sm, wp, ln, offset, readbuf, len);
      else
        cuda_api_write_local_memory (dev, sm, wp, ln, offset, writebuf, len);

      *xfered_len = len;
      return TARGET_XFER_OK;

    /* The stack lives in local memory for ABI compilations. */
    case TARGET_OBJECT_STACK_MEMORY:

      if (cuda_coords_get_current_physical (&dev, &sm, &wp, &ln))
        return TARGET_XFER_E_IO;
      if (readbuf)
        cuda_api_read_local_memory (dev, sm, wp, ln, offset, readbuf, len);
      else
        cuda_api_write_local_memory (dev, sm, wp, ln, offset, writebuf, len);

      *xfered_len = len;
      return status;

    /* When stopping on the device, build a simple siginfo object */
    case TARGET_OBJECT_SIGNAL_INFO:

      return cuda_nat_xfer_siginfo (ops, object, annex, readbuf, writebuf,
                                    offset, len, xfered_len);
  }

  /* Fallback to host routines for other types of memory objects */
  return host_target_ops.to_xfer_partial (ops, object, annex, readbuf,
                                          writebuf, offset, len,
					  xfered_len);
}

static void
cuda_nat_kill (struct target_ops *ops)
{
   /* XXX potential race condition here. we kill the application, and will later kill
      the device when finalizing the API. Should split the process into smaller steps:
      kill the device, then kill the app, then kill the dispatcher thread, then free
      resources in gdb (cuda_cleanup). OR do one initialize/finalize of the API per
      gdb run. */
  cuda_cleanup ();
  if (cuda_exception_is_valid (cuda_exception))
    cuda_exception_reset (cuda_exception);

  host_target_ops.to_kill (ops);
}

/* We don't want to issue a full mourn if we've encountered a cuda exception,
   because the host application has not actually reported that it has
   terminated yet. */
static void
cuda_nat_mourn_inferior (struct target_ops *ops)
{
  /* Mark breakpoints uninserted in case something tries to delete a
     breakpoint while we delete the inferior's threads (which would
     fail, since the inferior is long gone).  */
  mark_breakpoints_out ();

  cuda_cleanup ();
  if (cuda_exception_is_valid (cuda_exception))
    cuda_exception_reset (cuda_exception);

  host_target_ops.to_mourn_inferior (ops);
}

static bool sendAck = false;

/*This discusses how CUDA device exceptions are handled.  This
   includes hardware exceptions that are detected and propagated
   through the Debug API.

   Similarly, we optionally integrate CUDA Memcheck with CUDA GDB
   (set cuda memcheck on), to get memory access checking in CUDA code.

   We adopt the host semantics such that a device exception will
   terminate the host application as well. This is the simplest option
   for now. For this purpose we use the boolean cuda_exception_is_valid
   (cuda_exception) to track the propagation of this event.

   This is a 3-step process:

   1. cuda_wait ()

     Device exception detected.  We indicate this process has "stopped"
     (i.e. is not yet terminated) with a signal.  We suspend the
     device, and allow a user to inspect their program for the reason
     why they hit the fault.  cuda_exception_is_valid (cuda_exception) is set
     to true at this point.

   2. cuda_resume ()

     The user has already inspected any and all state in the
     application, and decided to resume.  On host systems, you cannot
     resume your app beyond a terminal signal (the app dies).  So,
     since in our case the app doesn't die, we need to enforce this if
     we desire the same behavior.  This is done by seeeing if
     cuda_exception_is_valid (cuda_exception) is set to true.

   3. cuda_wait ()

     If cuda_exception_is_valid (cuda_exception) is set, then we know we've
     killed the app due to an exception.  We need to indicate the process has
     been "signalled" (i.e. app has terminated) with a signal.  At this point,
     cuda_exception_is_valid (cuda_exception) is set back to false.  Process
     mourning ensues and the world is a better place.
*/

extern struct breakpoint *step_resume_breakpoint;

#if defined(__linux__) && defined(GDB_NM_FILE)
static void
cuda_clear_pending_sigint (pid_t pid)
{
  int status = 0, options = 0;
  ptrace (PTRACE_CONT, pid, 0, 0); /* Resume the host to consume the pending SIGINT */
  waitpid (pid, &status, options); /* Ensure we return for the right reason */
  gdb_assert (WIFSTOPPED (status) && WSTOPSIG (status) == SIGINT);
}
#endif

static int
cuda_check_pending_sigint (pid_t pid)
{
#if defined(__linux__) && defined(GDB_NM_FILE)
  sigset_t pending, blocked, ignored;

  linux_proc_pending_signals (pid, &pending, &blocked, &ignored);
  if (sigismember (&pending, SIGINT))
    {
      cuda_clear_pending_sigint (pid);
      return 1;
    }
#endif

   /* No pending SIGINT */
   return 0;
}

struct cuda_sigtrap_info_st {
  bool saved;
  uint32_t print;
  uint32_t stop;
} cuda_sigtrap_info;

void
cuda_sigtrap_set_silent (void)
{
  enum gdb_signal sigtrap = gdb_signal_from_host (SIGTRAP);

  if (cuda_options_stop_signal() != GDB_SIGNAL_TRAP) return;

  cuda_sigtrap_info.stop = signal_stop_state (sigtrap);
  cuda_sigtrap_info.print = signal_print_state (sigtrap);
  cuda_sigtrap_info.saved = true;

  signal_stop_update (sigtrap, 0);
  signal_print_update (sigtrap, 0);
}

void
cuda_sigtrap_restore_settings (void)
{
  enum gdb_signal sigtrap = gdb_signal_from_host (SIGTRAP);

  if (cuda_options_stop_signal() != GDB_SIGNAL_TRAP) return;

  if (cuda_sigtrap_info.saved)
    {
      signal_stop_update (sigtrap, cuda_sigtrap_info.stop);
      signal_print_update (sigtrap, cuda_sigtrap_info.print);
      cuda_sigtrap_info.saved = false;
    }
}

extern int cuda_host_want_singlestep;

/*CUDA_RESUME:

  For the meaning and interaction of ptid and sstep, read gnu-nat.c,
  line 1924.

  The actions of cuda_resume are based on 3 inputs: sstep, host_sstep
  and cuda_focus_is_device(). The actions are summarized in this
  table. 'sstep/resume dev' means single-stepping/resuming the device
  in focus if any, respectively.  'resume other dev' means resume any
  active device that is not in focus.

      device   sstep sstep | sstep   resume   resume    resume    sstep
      focus           host |  dev     dev    other dev   host      host
      ------------------------------------------------------------------
        0        0     0   |   0       1         1        1(b)      0
        0        0     1   |   0       0         0        1(c)      0
        0        1     0   |   -       -         -        -         -
        0        1     1   |   0       0         0        0         1
        1        0     0   |   0       1         1        1         0
        1        0     1   |   0       1         1        1         0
        1        1     0   |   1       0         0        0(a)      0
        1        1     1   |   1       0         0        0(a)      0

     (a) because we fake single-stepping to GDB by not calling the
     wait() routine, there is no need to resume the host. We used to
     resume the host so that the host could capture any SIGTRAP signal
     sent during single-stepping.

     (b) currently, there is no way to resume a single device, without
     resuming the rest of the world. That would lead to a deadlock.

     (c) In case host is resumed to simulate a single stepping,
     device should remain suspended.
*/
static void
cuda_do_resume (struct target_ops *ops, ptid_t ptid,
                     int sstep, int host_sstep, enum gdb_signal ts)
{
  uint32_t dev;

  cuda_sstep_reset (sstep);

  // Is focus on host?
  if (!cuda_focus_is_device())
    {
      // If not sstep - resume devices
      if (!host_sstep)
        for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
            device_resume (dev);

      // resume the host
      host_target_ops.to_resume (ops, ptid, sstep, ts);
      return;
    }

   // sstep the device
  if (sstep)
    {
      if (cuda_sstep_execute (inferior_ptid))
        return;
      /* If single stepping failed, plant a temporary breakpoint
         at the previous frame and resume the device */
      if (cuda_options_software_preemption ())
        {
          /* Physical coordinates might change even if API call has failed
           * if software preemption is enabled */
          int rc;
          uint64_t kernel_id, grid_id;
          CuDim3 block_idx, thread_idx;

          rc = cuda_coords_get_current_logical (&kernel_id, &grid_id, &block_idx, &thread_idx);
          if (rc)
            error (_("Failed to get current logical coordinates on GPU!"));
          /* Invalidate current coordinates as well as device cache */
          device_invalidate (cuda_current_device ());
          cuda_coords_invalidate_current ();

          rc = cuda_coords_set_current_logical (kernel_id, grid_id, block_idx, thread_idx);
          if (rc)
            error (_("Failed to find physical coordinates matching logical ones!"));
        }
      cuda_sstep_reset (false);
      insert_step_resume_breakpoint_at_caller (get_current_frame ());
      cuda_insert_breakpoints ();
    }

  // resume the device
  device_resume (cuda_current_device ());

  // resume other devices
  if (!cuda_notification_pending ())
    for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
      if (dev != cuda_current_device ())
        device_resume (dev);

  // resume the host
  host_target_ops.to_resume (ops, ptid, 0, ts);
}

static void
cuda_nat_resume (struct target_ops *ops, ptid_t ptid, int sstep, enum gdb_signal ts)
{
  uint32_t dev;
  cuda_coords_t c;
  bool cuda_event_found = false;
  int host_want_sstep = cuda_host_want_singlestep;
  CUDBGEvent event;

  cuda_trace ("cuda_resume: sstep=%d", sstep);
  cuda_host_want_singlestep = 0;

  if (!cuda_options_device_resume_on_cpu_dynamic_function_call () &&
      inferior_thread ()->control.in_infcall)
    {
      host_target_ops.to_resume (ops, ptid, 0, ts);
      return;
    }

  /* In cuda-gdb we have two types of device exceptions :
     Recoverable : CUDA_EXCEPTION_WARP_ASSERT
     Nonrecoverable : All others (e.g. CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS)

     The main difference is that a recoverable exception ensures that device
     state is consistent. Therefore, the user can request that the device
     continue execution. Currently, CUDA_EXCEPTION_WARP_ASSERT is the only
     recoverable exception.

     When a device side exception is hit, it sets cuda_exception in cuda_wait.
     In the case of a nonrecoverable exception, the cuda_resume call
     kills the host application and return early. The subsequent cuda_wait
     call cleans up the exception state.
     In the case of a recoverable exception, cuda-gdb must reset the exception
     state here and can then continue executing.
     In the case of CUDA_EXCEPTION_WARP_ASSERT, the handling of the
     exception (i.e. printing the assert message) is done as part of the
     cuda_wait call.
  */
  if (cuda_exception_is_valid (cuda_exception) &&
      !cuda_exception_is_recoverable (cuda_exception))
    {
      cuda_cleanup ();
      cuda_exception_reset (cuda_exception);
      host_target_ops.to_resume (ops, ptid, 0, GDB_SIGNAL_KILL);
      cuda_trace ("cuda_resume: exception found");
      return;
    }

  if (cuda_exception_is_valid (cuda_exception) &&
      cuda_exception_is_recoverable (cuda_exception))
    {
      cuda_exception_reset (cuda_exception);
      cuda_trace ("cuda_resume: recoverable exception found\n");
    }

  /* We have now handled all the CUDA notifications. We are ready to
     handle the next batch when the world resumes. Pending CUDA
     timeout events will be ignored until next time. */
  cuda_notification_mark_consumed ();
  cuda_sigtrap_restore_settings ();

  /* Check if a notification was received while a previous event was being
     serviced. If yes, check the event queue for a pending event, and service
     the event if one is found. */
  if (cuda_notification_aliased_event ())
    {
      cuda_notification_reset_aliased_event ();
      cuda_api_get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;

      if (cuda_event_found)
        {
          cuda_process_events (&event, CUDA_EVENT_SYNC);
          sendAck = true;
        }
    }

  /* Acknowledge the CUDA debugger API (for synchronous events) */
  if (sendAck)
    {
      cuda_api_acknowledge_sync_events ();
      sendAck = false;
    }

  cuda_do_resume (ops, ptid, sstep, host_want_sstep, ts);

  cuda_clock_increment ();
  cuda_trace ("cuda_resume: done");
}

/*CUDA_WAIT:

  The wait function freezes the world, update the cached information
  about the CUDA devices, and cook the wait_status.

  If we hit a SIGTRAP because of a debugger API event, qualify the
  signal as spurious, so that GDB ignores it.

  If we are single-stepping the device, we never resume the host. But
  GDB needs to believe a SIGTRAP has been received. We fake the
  target_wait_status accordingly. If we are stepping instruction
  (stepi, not step), GDB cannot guarantee that there is an actual next
  instruction (unlike stepping a source line). If the kernel dies, we
  have to recognize the case.

  Device exceptions (including memcheck violations) are presented to GDB as
  unique signals (defined in signal.[c,h]).  Everything else is presented
  as a SIGTRAP, spurious in the case of a debugger API event.
 */
static ptid_t
cuda_nat_wait (struct target_ops *ops, ptid_t ptid,
               struct target_waitstatus *ws,
               int target_options)
{
  ptid_t r;
  uint32_t dev, dev_id;
  uint64_t grid_id;
  kernel_t kernel;
  bool cuda_event_found = false;
  CUDBGEvent event, asyncEvent;
  struct thread_info *tp;
  cuda_coords_t c;

  cuda_trace ("cuda_wait");

  if (!cuda_options_device_resume_on_cpu_dynamic_function_call () &&
      inferior_thread ()->control.in_infcall)
    return host_target_ops.to_wait (ops, ptid, ws, target_options);

  if (cuda_exception_is_valid (cuda_exception))
    {
      ws->kind = TARGET_WAITKIND_SIGNALLED;
      ws->value.sig = (enum gdb_signal) cuda_exception_get_value (cuda_exception);
      cuda_exception_reset (cuda_exception);
      cuda_trace ("cuda_wait: exception found");
      return inferior_ptid;
    }
  else if (cuda_sstep_is_active ())
    {
      /* Cook the ptid and wait_status if single-stepping a CUDA device. */
      cuda_trace ("cuda_wait: single-stepping");
      r = cuda_sstep_ptid ();

      /* When stepping the device, the host process remains suspended.
         So, if the user issued a Ctrl-C, we wouldn't detect it since
         we never actually check its wait status.  We must explicitly
         check for a pending SIGINT here.
         if quit_flag is set then C-c was pressed in gdb session
         but signal was yet not forwarded to debugged process */
      if (cuda_check_pending_sigint (ptid_get_pid (r)) || check_quit_flag() )
        {
          ws->kind = TARGET_WAITKIND_STOPPED;
          ws->value.sig = GDB_SIGNAL_INT;
          cuda_set_signo (GDB_SIGNAL_INT);
        }
      else
        {
          ws->kind = TARGET_WAITKIND_STOPPED;
          ws->value.sig = GDB_SIGNAL_TRAP;
          cuda_set_signo (GDB_SIGNAL_TRAP);

          /* If we single stepped the last warp on the device, then the
             launch has completed.  However, we do not see the event for
             kernel termination until we resume the application.  We must
             explicitly handle this here by indicating the kernel has
             terminated and switching to the remaining host thread. */
          if (cuda_sstep_kernel_has_terminated ())
            {
              /* Only destroy the kernel that has been stepped to its exit */
              dev_id  = cuda_sstep_dev_id ();
              grid_id = cuda_sstep_grid_id ();
              kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
              kernels_terminate_kernel (kernel);

              /* Invalidate current coordinates and device state */
              cuda_coords_invalidate_current ();
              device_invalidate (dev_id);

              /* Consume any asynchronous events, if necessary.  We need to do
                 this explicitly here, since we're taking the quick path out of
                 this routine (and bypassing the normal check for API events). */
              cuda_api_get_next_async_event (&asyncEvent);
              if (asyncEvent.kind != CUDBG_EVENT_INVALID)
                cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

              /* Update device state/kernels */
              kernels_update_terminated ();
              cuda_update_convenience_variables ();

              switch_to_thread (r);
              tp = inferior_thread ();
              tp->control.step_range_end = 1;
              return r;
            }
        }
    }
  else
    {
      cuda_trace ("cuda_wait: host_wait\n");
      cuda_coords_invalidate_current ();
      r = host_target_ops.to_wait (ops, ptid, ws, target_options);

      /* GDB reads events asynchronously without blocking. The target may have
	 taken too long to reply and GDB did not get any events back.  Check if
	 this is the case and just return.  */
      if (ws->kind == TARGET_WAITKIND_IGNORE
	  || ws->kind == TARGET_WAITKIND_NO_RESUMED)
	return r;

      if (ws->kind == TARGET_WAITKIND_STOPPED &&
	  ws->value.sig == GDB_SIGNAL_0)
	{
	  /* GDB is trying to stop this thread.  Let it do it.  */
	  return r;
	}
    }

  /* Immediately detect if the inferior is exiting.
     In these situations, do not investigate the device. */
  if (ws->kind == TARGET_WAITKIND_EXITED)
    {
      cuda_trace ("cuda_wait: target is exiting, avoiding device inspection");
      return r;
    }

  /* Return if cuda has not been initialized yet */
  if (!cuda_initialize_target ())
    return r;

  /* Suspend all the CUDA devices. */
  cuda_trace ("cuda_wait: suspend devices");
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    device_suspend (dev);

  /* Check for asynchronous events.  These events do not require
     acknowledgement to the debug API, and may arrive at any time
     without an explicit notification. */
  cuda_api_get_next_async_event (&asyncEvent);
  if (asyncEvent.kind != CUDBG_EVENT_INVALID)
    cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

  /* Analyze notifications.  Only check for new events if we've
     we've received a notification, or if we're single stepping
     the device (since if we're stepping we wouldn't receive an
     explicit notification). */
  tp = inferior_thread ();
  cuda_notification_analyze (r, ws, tp->control.trap_expected);
  if (cuda_notification_received ())
    {
      /* Check if there is any CUDA event to be processed */
      cuda_api_get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;
    }

  /* Handle all the CUDA events immediately.  In particular, for
     GPU events that may happen without prior notification (GPU
     grid launches, for example), API events will be packed
     alongside of them, so we need to process the API event first. */
  if (cuda_event_found)
    {
      cuda_process_events (&event, CUDA_EVENT_SYNC);
      sendAck = true;
    }

  /* Update the info about the kernels */
  kernels_update_terminated ();

  /* Decide which thread/kernel to switch focus to. */
  if (cuda_exception_hit_p (cuda_exception))
    {
      cuda_trace ("cuda_wait: stopped because of an exception");
      c = cuda_exception_get_coords (cuda_exception);
      cuda_coords_set_current (&c);
      cuda_exception_print_message (cuda_exception);
      ws->kind = TARGET_WAITKIND_STOPPED;
      ws->value.sig = (enum gdb_signal) cuda_exception_get_value (cuda_exception);
      cuda_set_signo (cuda_exception_get_value (cuda_exception));
    }
  else if (cuda_sstep_is_active ())
    {
      cuda_trace ("cuda_wait: stopped because we are single-stepping");
      cuda_coords_update_current (false, false);
    }
  else if (cuda_breakpoint_hit_p (cuda_clock ()))
    {
      cuda_trace ("cuda_wait: stopped because of a breakpoint");
      /* Alias received signal to SIGTRAP when hitting a trap */
      cuda_set_signo (GDB_SIGNAL_TRAP);
      ws->value.sig = GDB_SIGNAL_TRAP;
      cuda_coords_update_current (true, false);
    }
  else if (cuda_system_is_broken (cuda_clock ()))
    {
      cuda_trace ("cuda_wait: stopped because there are broken warps (induced trap?)");
      /* Alias received signal to SIGTRAP when hitting a breakpoint */
      cuda_set_signo (GDB_SIGNAL_TRAP);
      ws->value.sig = GDB_SIGNAL_TRAP;
      cuda_coords_update_current (false, false);
    }
  else if (cuda_api_get_attach_state () == CUDA_ATTACH_STATE_APP_READY)
    {
      /* Finished attaching to the CUDA app.
         Preferably switch focus to a device if possible */
      struct inferior *inf = find_inferior_pid (ptid_get_pid (r));
      cuda_trace ("cuda_wait: stopped because we attached to the CUDA app");
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
      inf->control.stop_soon = STOP_QUIETLY;
      cuda_coords_update_current (false, false);
    }
  else if (cuda_api_get_attach_state () == CUDA_ATTACH_STATE_DETACH_COMPLETE)
    {
      /* Finished detaching from the CUDA app. */
      struct inferior *inf = find_inferior_pid (ptid_get_pid (r));
      cuda_trace ("cuda_wait: stopped because we detached from the CUDA app");
      inf->control.stop_soon = STOP_QUIETLY;
    }
  else if (ws->value.sig == GDB_SIGNAL_INT)
    {
      /* CTRL-C was hit. Preferably switch focus to a device if possible */
      cuda_trace ("cuda_wait: stopped because a SIGINT was received.");
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_coords_update_current (false, false);
    }
  else if (check_quit_flag ())
    {
      /* cuda-gdb received sigint, probably Nsight tries to stop the app. */
      cuda_trace ("cuda_wait: stopped because SIGINT was received by debugger.");
      ws->kind = TARGET_WAITKIND_STOPPED;
      ws->value.sig = GDB_SIGNAL_INT;
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_coords_update_current (false, false);
    }
  else if (cuda_event_found)
    {
      cuda_trace ("cuda_wait: stopped because of a CUDA event");
      cuda_sigtrap_set_silent ();
      cuda_coords_update_current (false, false);
    }
  else if (cuda_notification_received ())
    {
      /* No reason found when actual reason was consumed in a previous iteration (timeout,...) */
      cuda_trace ("cuda_wait: stopped for no visible CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP); /* Dummy signal. We stopped after all. */
      cuda_coords_invalidate_current ();
    }
  else
    {
      cuda_trace ("cuda_wait: stopped for a non-CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      cuda_coords_invalidate_current ();
    }

  cuda_adjust_host_pc (r);

  /* CUDA - managed memory */
  if (ws->kind == TARGET_WAITKIND_STOPPED &&
      (ws->value.sig == GDB_SIGNAL_BUS || ws->value.sig == GDB_SIGNAL_SEGV))
    {
      uint64_t addr = 0;
      struct gdbarch *arch = get_current_arch();
      int arch_ptr_size = gdbarch_ptr_bit (arch) / 8;
      LONGEST len = arch_ptr_size;
      LONGEST offset = arch_ptr_size == 8 ? 0x10 : 0x0c;
      LONGEST read = 0;
      gdb_byte *buf = (gdb_byte *)&addr;
      int inf_exec = is_executing (inferior_ptid);

      /* Mark inferior_ptid as not executing while reading object signal info*/
      set_executing (inferior_ptid, 0);
      read = target_read (&host_target_ops, TARGET_OBJECT_SIGNAL_INFO, NULL, buf, offset, len);
      set_executing (inferior_ptid, inf_exec);

      /* Check the results */
      if (read == len && cuda_managed_address_p (addr))
        {
          ws->value.sig = GDB_SIGNAL_CUDA_INVALID_MANAGED_MEMORY_ACCESS;
          cuda_set_signo (ws->value.sig);
        }
    }
  cuda_managed_memory_clean_regions();

  /* Switch focus and update related data */
  cuda_update_convenience_variables ();
  if (cuda_focus_is_device ())
    /* Must be last, once focus and elf images have been updated */
    switch_to_cuda_thread (NULL);

  cuda_trace ("cuda_wait: done");
  return r;
}

static void
cuda_nat_fetch_registers (struct target_ops *ops,
                          struct regcache *regcache,
                          int regno)
{
  uint64_t val;
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_regcache_arch (regcache);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  int num_regs = gdbarch_num_regs (gdbarch);

  /* delegate to the host routines when not on the device */
  if (!cuda_focus_is_device ())
    {
      host_target_ops.to_fetch_registers (ops, regcache, regno);
      return;
    }

  cuda_coords_get_current (&c);

  /* if all the registers are wanted, then we need the host registers and the
     device PC */
  if (regno == -1)
    {
      host_target_ops.to_fetch_registers (ops, regcache, regno);
      val = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
      regcache_raw_supply (regcache, pc_regnum, &val);
      return;
    }

  /* get the PC */
  if (regno == pc_regnum )
    {
      val = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
      regcache_raw_supply (regcache, pc_regnum, &val);
      return;
    }

  /* raw register */
  val = lane_get_register (c.dev, c.sm, c.wp, c.ln, regno);
  regcache_raw_supply (regcache, regno, &val);
}

static void
cuda_nat_store_registers (struct target_ops *ops,
                          struct regcache *regcache,
                          int regno)
{
  uint64_t val;
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_regcache_arch (regcache);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  int num_regs = gdbarch_num_regs (gdbarch);

  gdb_assert (regno >= 0 && regno < num_regs);

  if (!cuda_focus_is_device ())
    {
      host_target_ops.to_store_registers (ops, regcache, regno);
      return;
    }

  if (regno == pc_regnum)
    error (_("The PC of CUDA thread is not writable"));

  cuda_coords_get_current (&c);
  regcache_raw_collect (regcache, regno, &val);
  lane_set_register (c.dev, c.sm, c.wp, c.ln, regno, val);
}

static int
cuda_nat_insert_breakpoint (struct target_ops *ops, struct gdbarch *gdbarch,
			    struct bp_target_info *bp_tgt)
{
  uint32_t dev;
  bool inserted;

  gdb_assert (bp_tgt->owner != NULL ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_arm ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_aarch64);

  if (!bp_tgt->owner || !bp_tgt->owner->cuda_breakpoint)
    return host_target_ops.to_insert_breakpoint (ops, gdbarch, bp_tgt);

  /* Insert the breakpoint on whatever device accepts it (valid address). */
  inserted = false;
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    {
      inserted |= cuda_api_set_breakpoint (dev, bp_tgt->reqstd_address);
    }

  /* Make sure we save the address where the actual breakpoint was placed.  */
  if (inserted)
    bp_tgt->placed_address = bp_tgt->reqstd_address;

  return !inserted;
}

static int
cuda_nat_remove_breakpoint (struct target_ops *ops, struct gdbarch *gdbarch,
			    struct bp_target_info *bp_tgt, enum remove_bp_reason reason)
{
  uint32_t dev;
  CORE_ADDR cuda_addr;
  bool removed;

  gdb_assert (bp_tgt->owner != NULL ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_arm ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_aarch64);

  if (!bp_tgt->owner || !bp_tgt->owner->cuda_breakpoint)
    return host_target_ops.to_remove_breakpoint (ops, gdbarch, bp_tgt, reason);

  /* Removed the breakpoint on whatever device accepts it (valid address). */
  removed = false;
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    {
      /* We need to remove breakpoints even if no kernels remain on the device */
      removed |= cuda_api_unset_breakpoint (dev, bp_tgt->placed_address);
    }
  return !removed;
}

struct gdbarch *
cuda_nat_thread_architecture (struct target_ops *ops, ptid_t ptid)
{
  if (cuda_focus_is_device ())
    return cuda_get_gdbarch ();
  else
    return target_gdbarch();
}

void
cuda_focus_init (cuda_focus_t *focus)
{
  gdb_assert (focus);

  focus->valid = false;
}

void
cuda_focus_save (cuda_focus_t *focus)
{
  gdb_assert (focus);
  gdb_assert (!focus->valid);

  focus->ptid = inferior_ptid;
  focus->coords = CUDA_INVALID_COORDS;
  cuda_coords_get_current (&focus->coords);

  focus->valid = true;
}

void
cuda_focus_restore (cuda_focus_t *focus)
{
  gdb_assert (focus);
  gdb_assert (focus->valid);

  if (focus->coords.valid)
    switch_to_cuda_thread  (&focus->coords);
  else
    switch_to_thread (focus->ptid);

  focus->valid = false;
}

void
switch_to_cuda_thread (cuda_coords_t *coords)
{
  cuda_coords_t c;

  gdb_assert (coords || cuda_focus_is_device ());

  if (coords)
    cuda_coords_set_current (coords);
  cuda_coords_get_current (&c);

  cuda_update_cudart_symbols ();
  reinit_frame_cache ();
  registers_changed ();
  stop_pc = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
}


static struct objfile *cuda_create_builtins_objfile (void);

void
cuda_update_cudart_symbols (void)
{
  /* If not done yet, create a CUDA runtime symbols file */
  if (!cuda_cudart_symbols.objfile)
    {
      cuda_cudart_symbols.objfile = cuda_create_builtins_objfile ();
    }

}

void
cuda_cleanup_cudart_symbols (void)
{
  /* Free the objfile if allocated */
  if (cuda_cudart_symbols.objfile)
    {
      free_objfile (cuda_cudart_symbols.objfile);
      cuda_cudart_symbols.objfile = NULL;
    }
}

#if defined(__APPLE__)
/* CUDA - cuda-gdb wrapper */
/* Smuggle the DYLD_* environement variables like GDB 6.3.5 used to do. Because
   we cuda-gdb must be part of the procmod group, those variables are not
   passed, for security reasons. Instead we pass them in a GDB_DYLD_*
   equivalent and restore them when launching the inferior. This code was taken
   from GDB 6.3.5, the Apple editon. */ 

struct dyld_smuggle_pairs {
  const char *real_name;
  const char *smuggled_name;
};

static void
smuggle_dyld_settings (struct gdb_environ *e)
{
  int i;
  struct dyld_smuggle_pairs env_names[] = { 
       {"DYLD_FRAMEWORK_PATH",          "GDB_DYLD_FRAMEWORK_PATH"},
       {"DYLD_FALLBACK_FRAMEWORK_PATH", "GDB_DYLD_FALLBACK_FRAMEWORK_PATH"},
       {"DYLD_LIBRARY_PATH",            "GDB_DYLD_LIBRARY_PATH"},
       {"DYLD_FALLBACK_LIBRARY_PATH",   "GDB_DYLD_FALLBACK_LIBRARY_PATH"},
       {"DYLD_ROOT_PATH",               "GDB_DYLD_ROOT_PATH"},
       {"DYLD_PATHS_ROOT",              "GDB_DYLD_PATHS_ROOT"},
       {"DYLD_IMAGE_SUFFIX",            "GDB_DYLD_IMAGE_SUFFIX"},
       {"DYLD_INSERT_LIBRARIES",        "GDB_DYLD_INSERT_LIBRARIES"},
       { NULL,                          NULL } };

  for (i = 0; env_names[i].real_name != NULL; i++)
    {
      const char *real_val     = get_in_environ (e, env_names[i].real_name);
      const char *smuggled_val = get_in_environ (e, env_names[i].smuggled_name);

      if (real_val == NULL && smuggled_val == NULL)
        continue;

      if (smuggled_val == NULL)
        continue;

      /* Is the value of the DYLD_* env var truncated to ""? */
      if (real_val != NULL && real_val[0] != '\0')
        continue;

      /* real_val has a value and it looks legitimate - don't overwrite it
         with the smuggled version.  */
      if (real_val != NULL)
        continue;

      set_in_environ (e, env_names[i].real_name, smuggled_val);
    }
}
#endif

/* If a host event is hit while there are valid threads
   on the GPU, the focus ends up being switched to the 
   GPU, leaving the host PC not rewound.

   This function determines if the host is at a breakpoint,
   and if so it manually rewinds the host PC so that the
   breakpoint can be hit again after a resume.
   r here is the return value of host_wait().
*/
void 
cuda_adjust_host_pc (ptid_t r)
{
  bool pc_rewound = false;
  struct regcache *regcache;
  CORE_ADDR pc;
  cuda_coords_t coords;

  if (!cuda_focus_is_device ())
    return;

  /* Rewind host PC and consume pending SIGTRAP
     Sometimes, one thread can hit both a host and a device
     breakpoint at the same time, in which case host SIGTRAP
     is triggered while SIGTRAP from back end is blocked (pending).
     When resuming, host PC is not rewound because focus is on the
     device.

     Before switching to CUDA thread, we check if that's the case.
     If so, manually rewind the host PC and consume the pending SIGTRAP.
     This allows the host breakpoint to be hit again after resuming. */

  /* r is guaranteed to be the return of host_wait in this case */
  cuda_coords_get_current (&coords);

  /* Temporarily invalidate the current coords so that the focus
     is set on the host. */
  cuda_coords_invalidate_current ();

  regcache = get_thread_arch_regcache (r, target_gdbarch());
  pc = regcache_read_pc (regcache) - gdbarch_decr_pc_after_break (target_gdbarch());
  if (breakpoint_inserted_here_p (get_regcache_aspace (regcache), pc))
    {
        /* Rewind the PC */
      regcache_write_pc (regcache, pc);
      pc_rewound = true;
    }

  /* Restore coords */
  cuda_coords_set_current (&coords);

  /* Remove the pending notification if we rewound the pc */
  if (pc_rewound)
      cuda_notification_consume_pending ();
}

void
cuda_set_environment (struct gdb_environ *environ)
{
#if defined(__APPLE__)
  /* CUDA - cuda-gdb wrapper */
  smuggle_dyld_settings (environ);
#endif
}

/* The host target is overriden by calling when calling cuda_nat_add_target().
   The original is saved into host_target_ops because the cuda_nat target_ops
   will call the host functions when dealing with host requests */
void
cuda_nat_add_target (struct target_ops *t)
{
  static char shortname[128];
  static char longname[128];
  static char doc[256];

  if (!cuda_debugging_enabled)
    return;

  /* Save the original set of target operations */
  host_target_ops = *t;

  /* Build the meta-data strings without using malloc */
  strncpy (shortname, t->to_shortname, sizeof (shortname) - 1);
  strncat (shortname, " + cuda", sizeof (shortname) - 1 - strlen (shortname));
  strncpy (longname, t->to_longname, sizeof (longname) - 1);
  strncat (longname, " + CUDA support", sizeof (longname) - 1 - strlen (longname));
  strncpy (doc, t->to_doc, sizeof (doc) - 1);
  strncat (doc, " with CUDA support", sizeof (doc) - 1 - strlen (doc));

  /* Override what we need to */
  t->to_shortname             = shortname;
  t->to_longname              = longname;
  t->to_doc                   = doc;
  t->to_kill                  = cuda_nat_kill;
  t->to_mourn_inferior        = cuda_nat_mourn_inferior;
  t->to_resume                = cuda_nat_resume;
  t->to_wait                  = cuda_nat_wait;
  t->to_fetch_registers       = cuda_nat_fetch_registers;
  t->to_store_registers       = cuda_nat_store_registers;
  t->to_insert_breakpoint     = cuda_nat_insert_breakpoint;
  t->to_remove_breakpoint     = cuda_nat_remove_breakpoint;
  t->to_xfer_partial          = cuda_nat_xfer_partial;
  t->to_thread_architecture   = cuda_nat_thread_architecture;
  t->to_detach                = cuda_nat_detach;
}

bool cuda_debugging_enabled = false;

/* Provide a prototype to silence -Wmissing-prototypes.  */
extern initialize_file_ftype _initialize_cuda_nat;


static bool
cuda_get_cudbg_api (void)
{
  CUDBGAPI api = NULL;
  CUDBGResult res;


  res = cudbgGetAPI (CUDBG_API_VERSION_MAJOR,
                     CUDBG_API_VERSION_MINOR,
                     CUDBG_API_VERSION_REVISION,

                     &api);
  if (res == CUDBG_SUCCESS)
    cuda_api_set_api (api);

  cuda_api_handle_get_api_error (res);

  return (res != CUDBG_SUCCESS);
}

void
_initialize_cuda_nat (void)
{
  struct target_ops *t = NULL;

  /* Check the required CUDA debugger files are present */
  if (cuda_get_cudbg_api ())
    {
      warning ("CUDA support disabled: could not obtain the CUDA debugger API\n");
      cuda_debugging_enabled = false;
      return;
    }

  /* Initialize the CUDA modules */
  cuda_commands_initialize ();
  cuda_options_initialize ();
  cuda_notification_initialize ();

  /* Initialize the cleanup routines */
  make_final_cleanup (cuda_final_cleanup, NULL);

  cuda_debugging_enabled = true;
}

void
cuda_nat_attach (void)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  char *cudbgApiAttach = "cudbgApiAttach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR resumeAppOnAttachFlagAddr;
  unsigned char resumeAppOnAttach;
  unsigned int timeOut = 5000000; /* 5 seconds */
  unsigned int timeElapsed = 0;
  unsigned dev = 0;
  const unsigned int sleepTime = 1000;
  uint64_t internal_error_code;
  struct cleanup *cleanup = NULL;

  /* Return early if CUDA driver isn't available. Attaching to the host
     process has already been completed at this point. */
  cuda_api_set_attach_state (CUDA_ATTACH_STATE_IN_PROGRESS);
  if (!cuda_initialize_target ())
    {
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
      return;
    }

   /* If the CUDA driver has been loaded but software preemption has been turned
      on, stop the attach process. */
   if (cuda_options_software_preemption ())
    {
       cuda_api_set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
       error (_("Attaching to a running CUDA process with software preemption "
                "enabled in the debugger is not supported."));
    }


  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_("Failed to initiate attach."));

  /* Fork off the CUDA debugger process from the inferior */
  cleanup = cuda_gdb_bypass_signals ();
  cmd_func (cmd, cudbgApiAttach, 0);
  do_cleanups (cleanup);

  internal_error_code = cuda_get_last_driver_internal_error_code();

  if ((unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
     error (_("Attaching not possible. "
              "Please verify that software preemption is disabled "
              "and that nvidia-cuda-mps-server is not running."));
  if ((unsigned int)internal_error_code == CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED)
     error (_("Attaching to process running on watchdogged GPU is not possible.\n"
              "Please repeat the attempt in console mode or "
              "restart the process with CUDA_VISIBLE_DEVICES environment variable set."));
  if (internal_error_code)
    error (_("Attach failed due to the internal driver error 0x%llx\n"),
            (unsigned long long) internal_error_code);

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  resumeAppOnAttachFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_RESUME_FOR_ATTACH_DETACH));

  /* If this is not available, the CUDA driver doesn't support attaching.  */
  if (resumeAppOnAttachFlagAddr == 0 || debugFlagAddr == 0)
    error (_("This CUDA driver does not support attaching to a running CUDA process."));

  /* Wait till the backend has started up and is ready to service API calls */
  while (cuda_api_initialize () != CUDBG_SUCCESS)
    {
      internal_error_code = cuda_get_last_driver_internal_error_code();

      if ((unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
         error (_("Attaching not possible. "
                  "Please verify that software preemption is disabled "
                  "and that nvidia-cuda-mps-server is not running."));
      if (internal_error_code)
        error (_("Attach failed due to the internal driver error 0x%llx\n"),
                (unsigned long long) internal_error_code);

      if (timeElapsed < timeOut)
        usleep(sleepTime);
      else
        error (_("Timed out waiting for the CUDA API to initialize."));

      timeElapsed += sleepTime;
    }

  /* Check if the inferior needs to be resumed */
  target_read_memory (resumeAppOnAttachFlagAddr, &resumeAppOnAttach, 1);

  if (resumeAppOnAttach)
    {
      int  cnt;
      cleanup = cuda_gdb_bypass_signals ();
      /* Resume the inferior to collect more data. CUDA_ATTACH_STATE_COMPLETE and
         CUDBG_IPC_FLAG_NAME will be set once this completes. */
      for (cnt=0;
	   cnt < 1000
           && cuda_api_get_attach_state () == CUDA_ATTACH_STATE_IN_PROGRESS;
           cnt++)
	  {
	      prepare_execution_command (&current_target, true);
	      continue_1 (false);
	      wait_for_inferior ();
	      normal_stop ();
	  }

      /* No threads are running at this point.  */
      set_running (minus_one_ptid, 0);

      do_cleanups (cleanup);
      if (cuda_api_get_attach_state () != CUDA_ATTACH_STATE_APP_READY &&
          cuda_api_get_attach_state () != CUDA_ATTACH_STATE_COMPLETE)
          error ("Unexpected CUDA attach state, further debugging session might be unreliable");
    }
  else
    {
      /* Enable debugger callbacks from the CUDA driver */
      cuda_write_bool (debugFlagAddr, true);

      /* No data to collect, attach complete. */
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
    }

  /* Initialize CUDA and suspend the devices */
  cuda_initialize ();
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    device_suspend (dev);

  /* The inferior just got signaled, we're not expecting any other stop */
  current_inferior ()->control.stop_soon = NO_STOP_QUIETLY;
}

void cuda_do_detach(bool remote)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  char *cudbgApiDetach = "cudbgApiDetach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR resumeAppOnDetachFlagAddr;
  unsigned char resumeAppOnDetach;
  struct cleanup *cleanup = NULL;
  struct thread_info *tp;
  int pid;

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));

  /* Bail out if the CUDA driver isn't available */
  if (!debugFlagAddr)
      return;

  /* Update the suspended devices mask in the inferior */
  cuda_inferior_update_suspended_devices_mask ();

  cuda_api_set_attach_state (CUDA_ATTACH_STATE_DETACHING);

  /* Make sure the focus is set on the host */
  switch_to_thread (inferior_ptid);

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_("Failed to initiate detach."));

  /* Figure out if we need to clean up driver state before detaching */
  resumeAppOnDetachFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_RESUME_FOR_ATTACH_DETACH));

  if (!resumeAppOnDetachFlagAddr)
    error (_("Failed to detach cleanly from the inferior."));

  /* Make dynamic call for cleanup. */
  cleanup = cuda_gdb_bypass_signals ();
  cmd_func (cmd, cudbgApiDetach, 0);
  do_cleanups (cleanup);

  /* Read the updated value of the flag */
  target_read_memory (resumeAppOnDetachFlagAddr, &resumeAppOnDetach, 1);

  /* If this flag is set, the debugger backend needs to be notified to cleanup on detach */
  if (resumeAppOnDetach)
    cuda_api_request_cleanup_on_detach (resumeAppOnDetach);

  /* Make sure the debugger is reinitialized from scratch on reattaching
     to the inferior */
  rpcFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_DEBUGGER_INITIALIZED));

  if (!rpcFlagAddr)
    error (_("Failed to detach cleanly from the inferior."));

  cuda_write_bool (rpcFlagAddr, false);

  /* If a cleanup is needed, resume the app to allow the cleanup to complete.
     The debugger backend will send a cleanup event to stop the app when the
     cleanup finishes. */
  if (resumeAppOnDetach)
    {
      int cnt;

      /* Clear all breakpoints */
      delete_command (NULL, 0);
      cuda_system_cleanup_breakpoints ();
      cuda_options_disable_break_on_launch ();

      /* Now resume the app and wait for CUDA_ATTACH_STATE_DETACH_COMPLETE event. */
      for (cnt=0;
	   cnt < 100
           && cuda_api_get_attach_state () != CUDA_ATTACH_STATE_DETACH_COMPLETE;
           cnt++)
	  {
	      prepare_execution_command (&current_target, true);
	      continue_1 (false);
	      wait_for_inferior ();
	      normal_stop ();
	  }

      /* No threads are running at this point.  */
      set_running (minus_one_ptid, 0);
    }
  else
    cuda_api_set_attach_state (CUDA_ATTACH_STATE_DETACH_COMPLETE);

  if (cuda_api_get_attach_state () != CUDA_ATTACH_STATE_DETACH_COMPLETE)
    warning (_("Unexpected CUDA API attach state."));

  cuda_write_bool (debugFlagAddr, false);

  cuda_cleanup ();
}

static void
cuda_nat_detach (struct target_ops *ops, const char *args, int from_tty)
{
  /* If the Debug API is not initialized,
   * treat the inferior as a host-only process */
  if (cuda_api_get_state () == CUDA_API_STATE_INITIALIZED)
    cuda_do_detach (false);

  /* Do not try to detach from an already dead process */
  if (ptid_get_pid (inferior_ptid) == 0) return;

  /* Call the host detach routine. */
  host_target_ops.to_detach (ops, args, from_tty);
}

/*
 * CUDA builtins construction routines
 */

/* cuda_alloc_dim3_type helper routine: initializes one of the structure fields
 * with a given name, offset and type */
static void
cuda_init_field (struct field *fp, const char *name, const int offs, struct type *type)
{
  fp->name = name;
  fp->type = type;
  SET_FIELD_BITPOS(*fp, offs*8);
  FIELD_BITSIZE(*fp) = 16;
}


/* Allocates dim3 type as structure of 3 packed unsigned shorts: x, y and z */
static struct type *
cuda_alloc_dim3_type (struct objfile *objfile)
{
  struct gdbarch *gdbarch = get_objfile_arch (objfile); 
  struct type *uint32_type = builtin_type (gdbarch)->builtin_unsigned_int;
  struct type *dim3 = NULL;

  dim3 = alloc_type (objfile);

  TYPE_NAME(dim3) = "dim3";
  TYPE_LENGTH(dim3) = 12;
  TYPE_CODE(dim3) = TYPE_CODE_STRUCT;

  TYPE_NFIELDS(dim3) = 3;
  TYPE_FIELDS(dim3) = (struct field *)TYPE_ALLOC(dim3, sizeof (struct field) * 3);

  cuda_init_field (&TYPE_FIELD(dim3, 0), "x", 0, uint32_type);
  cuda_init_field (&TYPE_FIELD(dim3, 1), "y", 4, uint32_type);
  cuda_init_field (&TYPE_FIELD(dim3, 2), "z", 8, uint32_type);

  return dim3;
}


/* Create symbol of a given type inside the objfile */
static struct symbol *
cuda_create_symbol (struct objfile *objfile, const char *name, CORE_ADDR addr, struct type *type)
{
  struct symbol *sym = NULL;
  struct gdbarch *gdbarch = get_objfile_arch (objfile);
  const struct blockvector *bv = COMPUNIT_BLOCKVECTOR (objfile->compunit_symtabs);

  /* Allocate a new symbol in OBJFILE's obstack.  */
  sym = allocate_symbol (objfile);

  SYMBOL_SET_LANGUAGE (sym, language_c, &objfile->per_bfd->storage_obstack);
  SYMBOL_SET_NAMES (sym, name, strlen (name), 0, objfile);
  SYMBOL_TYPE (sym) = type;
  SYMBOL_DOMAIN (sym) = VAR_DOMAIN;
  SYMBOL_ACLASS_INDEX (sym) = LOC_STATIC;
  SYMBOL_VALUE_ADDRESS (sym) = addr;

  /* Register symbol as global symbol with symtab */
  symbol_set_symtab (sym, COMPUNIT_FILETABS (objfile->compunit_symtabs));
  add_symbol_to_list (sym, &global_symbols);
  dict_add_symbol (BLOCK_DICT (BLOCKVECTOR_BLOCK (bv, GLOBAL_BLOCK)), sym);

  return sym;
}

/* Symtab initialization helper routine:
 * Allocates blockvector as well as global and static blocks inside symtab 
 */
static void
cuda_alloc_blockvector (struct symtab *symtab)
{
  struct obstack *obstack = &(SYMTAB_OBJFILE (symtab)->objfile_obstack);
  struct blockvector *bv;
  struct block *bl;

  bv = (struct blockvector *) obstack_alloc (obstack, sizeof(struct blockvector) + sizeof(struct block *));
  BLOCKVECTOR_NBLOCKS (bv) = 1;

  /* Allocate global block */
  bl = allocate_global_block (obstack);
  BLOCK_DICT (bl) = dict_create_hashed_expandable ();
  BLOCKVECTOR_BLOCK (bv, GLOBAL_BLOCK) = bl;
  set_block_compunit_symtab (bl, SYMTAB_COMPUNIT (symtab));

  /* Allocate static block*/
  bl = allocate_global_block (obstack);
  BLOCK_DICT (bl) = dict_create_hashed_expandable ();
  BLOCKVECTOR_BLOCK (bv, STATIC_BLOCK) = bl;
  set_block_compunit_symtab (bl, SYMTAB_COMPUNIT (symtab));

  SYMTAB_BLOCKVECTOR (symtab) = bv;
}

/* Allocate virtual objfile and construct the following symbols inside it:
 * threadIdx of type dim3 located at CUDBG_THREADIDX_OFFSET
 * blockIdx of type dim3 located at CUDBG_BLOCKIDX_OFFSET
 * blockDim of type dim3 located at CUDBG_BLOCKDIM_OFFSET
 * threadDim of type dim3 located at CUDBG_THREADDIM_OFFSET
 * warpSize of type int located at CUDBG_WARPSIZE_OFFSET
 */
static struct objfile *
cuda_create_builtins_objfile (void)
{
  struct objfile *objfile = NULL;
  struct type *int32_type = NULL;
  struct type *dim3_type = NULL;
  struct symtab *symtab = NULL;
  struct blockvector *bv = NULL;
  struct cleanup *cleanups = NULL;

  /* Initialized things since we are going to start assembling a new
     objfile and reading symbols.  */
  buildsym_init ();

  /* Set the cleanup chain so we get things properly set after we're done
     assembling the symbol table.  */
  cleanups = make_cleanup (really_free_pendings, NULL);

  /* This is not a real objfile.  Mark it as so by passing
     OBJF_NOT_FILENAME.  */
  objfile = allocate_objfile (NULL, NULL, OBJF_SHARED | OBJF_NOT_FILENAME);
  objfile->per_bfd->gdbarch = cuda_get_gdbarch ();

  /* Get/allocate types */
  int32_type = builtin_type (get_objfile_arch (objfile))->builtin_int32;
  dim3_type = cuda_alloc_dim3_type (objfile);

  /* Now that the objfile structure has been allocated, we need to allocate all
     the required data structures for symbols.  */
  objfile->compunit_symtabs = allocate_compunit_symtab (objfile, "<cuda-builtins>");

  symtab = allocate_symtab (objfile->compunit_symtabs, "<cuda-builtins>");

  symtab->language = language_c;

  cuda_alloc_blockvector (symtab);

  cuda_create_symbol (objfile, "threadIdx", CUDBG_THREADIDX_OFFSET, dim3_type);
  cuda_create_symbol (objfile, "blockIdx", CUDBG_BLOCKIDX_OFFSET, dim3_type);
  cuda_create_symbol (objfile, "blockDim", CUDBG_BLOCKDIM_OFFSET, dim3_type);
  cuda_create_symbol (objfile, "gridDim", CUDBG_GRIDDIM_OFFSET, dim3_type);
  cuda_create_symbol (objfile, "warpSize", CUDBG_WARPSIZE_OFFSET, int32_type);

  do_cleanups (cleanups);

  return objfile;
}
