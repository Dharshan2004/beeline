from __future__ import annotations

import sys


def peak_rss_bytes() -> int | None:
    """Peak resident set size for this process, or None if unavailable.

    Reported alongside build and load timings so the memory cost of the dense
    Retrieval Route can be weighed against the packaged-size budget.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        # Without an explicit restype the HANDLE is truncated to a C int and the
        # query silently fails on 64-bit Python.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)

    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return int(usage) if sys.platform == "darwin" else int(usage) * 1024
