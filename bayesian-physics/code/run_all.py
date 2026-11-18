"""Run every worked example in the course and report any that fail."""
import glob, runpy, sys, os
here = os.path.dirname(os.path.abspath(__file__))
failed = []
for path in sorted(glob.glob(os.path.join(here, "[0-9][0-9]_*.py"))):
    print("\n" + "=" * 78 + "\n" + os.path.basename(path) + "\n" + "=" * 78)
    try:
        runpy.run_path(path, run_name="__main__")
    except Exception as err:
        failed.append((os.path.basename(path), repr(err)))
print("\nfinished;", "all files ran" if not failed else "FAILED: %s" % failed)
sys.exit(1 if failed else 0)
