import os, shlex, shutil, subprocess, time, socket
from paraview.simple import Connect, servermanager, Wavelet, Delete     # type: ignore
from vtkmodules.vtkParallelCore import vtkMultiProcessController    # type: ignore


def activate_local_pvserver(np: int, server_port: int, timeout_s: float) -> bool:
    """
    Launch pvserver in a new Linux terminal and verify it is listening.

    Args:
        np (int): Number of MPI processes (>= 1). If 1, run pvserver directly; if >1, use mpiexec -n <np>.
        server_port (int): TCP port for pvserver --server-port (1..65535).
        timeout_s (float): Seconds to wait for the port to start listening.

    Returns:
        bool: True if the server appears to be up (port is listening), False otherwise.
    """
    import os, shlex, shutil, subprocess, time

    #### Validate inputs
    if not isinstance(np, int) or np < 1:
        raise ValueError("np must be an integer >= 1")
    if not (1 <= server_port <= 65535):
        raise ValueError("server_port must be in [1, 65535]")

    ### Build the command
    # (0) force offscreen + backtrace flags
    base_cmd = ["pvserver", "--server-port", str(server_port), "--force-offscreen-rendering", "--enable-bt"]

    # Single thread option
    if np == 1:
        cmd = ["env", "LIBGL_ALWAYS_SOFTWARE=1", "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe"] + base_cmd
    # Multi-threaded option
    else:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if not mpiexec:
            raise FileNotFoundError("mpiexec/mpirun not found on PATH for np > 1")
        # MPICH/Hydra uses -l to label output
        cmd = [mpiexec, "-n", str(np), "-l",
               "env", "LIBGL_ALWAYS_SOFTWARE=1", "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe"] + base_cmd

    ### Pick a terminal emulator
    term = (shutil.which("gnome-terminal") or shutil.which("xfce4-terminal")
            or shutil.which("konsole") or shutil.which("mate-terminal")
            or shutil.which("xterm"))

    ### Prepare terminal launch command (runs pvserver inside a shell)
    if term:
        t = os.path.basename(term)
        cmd_str = shlex.join(cmd)

        # common bash tail: close on success, hold on error
        # - se rc == 0 → exit subito (il terminale si chiude)
        # - se rc != 0 → mostra exit code e aspetta
        bash_tail = (
            'rc=$?; '
            'if [ $rc -ne 0 ]; then '
            '  echo; echo "pvserver exit code: $rc"; '
            '  read -p "Press Enter to close..."; '
            'fi'
        )

        if t == "gnome-terminal":
            launch = [
                term, "--", "bash", "-lc",
                f'{cmd_str}; {bash_tail}'
            ]
        elif t == "xfce4-terminal":
            # xfce4-terminal con --hold terrebbe SEMPRE aperto: lo togliamo
            launch = [
                term, "-e",
                "bash -lc " + shlex.join([f'{cmd_str}; {bash_tail}'])
            ]
        elif t == "konsole":
            launch = [
                term, "-e", "bash", "-lc",
                f'{cmd_str}; {bash_tail}'
            ]
        elif t == "mate-terminal":
            launch = [
                term, "--", "bash", "-lc",
                f'{cmd_str}; {bash_tail}'
            ]
        else:  # xterm
            # xterm non ha il close-on-success nativo, quindi facciamo lo stesso trucco
            launch = [
                term, "-e", "bash", "-lc",
                f'{cmd_str}; {bash_tail}'
            ]

        subprocess.Popen(launch)

    else:
        # Fallback: run without opening a terminal (still works)
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ### Verify that the server is listening on the port
    def _is_listening_no_connect(port: int) -> bool:
        # Prefer 'ss' (does not create a connection)
        ss = shutil.which("ss")
        if ss:
            try:
                out = subprocess.run([ss, "-H", "-ltn"], capture_output=True, text=True, check=False).stdout
                # Look for a LISTEN line ending with :<port>
                return any(("LISTEN" in line) and (f":{port} " in line or line.rstrip().endswith(f":{port}"))
                           for line in out.splitlines())
            except Exception:
                pass
        # Fallback to lsof (also non-intrusive)
        lsof = shutil.which("lsof")
        if lsof:
            try:
                res = subprocess.run([lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                                     capture_output=True, text=True, check=False)
                return res.returncode == 0 and str(port) in res.stdout
            except Exception:
                pass
        # Last resort: just assume it's up (do NOT dial the port to avoid killing pvserver)
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_listening_no_connect(server_port):
            return True
        time.sleep(0.25)
    return False


def connect_to_local_pvserver(server_port: int) -> bool:
    """
    Connect to a set up pvserver and verify the connection.

    Args:
        server_port (int): TCP port for pvserver --server-port (1..65535).

    Returns:
        bool: True if the connection is established, False otherwise.
    """

    #### Validate inputs
    if not (1 <= server_port <= 65535):
        raise ValueError("server_port must be in [1, 65535]")

    ### Connect to the local server
    Connect("localhost", server_port)     

    ### Check if an active connection is detected
    if servermanager.ActiveConnection is None:
        print("[Connect] no ActiveConnection after Connect()")
        return False
    
    ### Handshake check (create test source and delete it)
    try:
        w = Wavelet()           # created on the server
        Delete(w)               # delete again
        return True
    except Exception as e:
        print(f"[Connect] handshake proxy failed: {e}")
        return False
    

# def mpi_barrier() -> None:
#     """
#     Synchronizes all mpi processes
#     """

#     try:
#         ctrl = vtkMultiProcessController.GetGlobalController()
#         if ctrl: ctrl.Barrier()
#     except Exception:
#         pass


# def mpi_is_lead_thread() -> bool:
#     """
#     Returns true for the thread with rank 0, False for all the others.
#     """

#     try:
#         ctrl = vtkMultiProcessController.GetGlobalController()
#         rank = ctrl.GetLocalProcessId() if ctrl else 0

#         if rank == 0:
#             return True
#         else:
#             return False
#     except Exception:
#         return True