import os
import shutil
import glob
from datetime import datetime

usb_dir = "/mnt/d/"  # assumes I have set up the mounting of Vanja USB stick (which allows Asus laptop to read SD card) at /mnt/d on WSL, if this isn't working, try remounting with mkdir /mnt/d ; sudo mount -t drvfs D: /mnt/d
ehd_dir = "/mnt/e/Audio"

extensions = [".WAV", ".hprj"]

# the Zoom H5 organizes them under either MULTI or STEREO, then as FOLDER01 or FOLDER02 etc., then as a project-specific folder e.g. ZOOM0001, where there will be the recording's ZOOM0001.WAV and ZOOM0001.hprj

for large_dir in ["MULTI", "STEREO"]:
    large_dir = os.path.join(usb_dir, large_dir)
    # print(f"now in {large_dir}")
    subdirs = [f.name for f in os.scandir(large_dir) if f.is_dir()]
    # use f.name for just immediate name e.g. FOLDER0001, use f.path for full path
    for subdir in subdirs:
        # these are the FOLDER01 etc.
        assert subdir.startswith("FOLDER"), subdir
        subdir = os.path.join(large_dir, subdir)
        # print(f"now in {subdir}")
        project_dirs = [f.name for f in os.scandir(subdir) if f.is_dir()]
        for project_dir in project_dirs:
            assert project_dir.startswith("ZOOM"), project_dir
            project_name = project_dir
            project_dir = os.path.join(subdir, project_name)
            # print(f"now in {project_dir}")

            date = datetime.utcnow().strftime("%Y%m%d")
            new_project_name = f"{date}-{project_name}"

            if not os.path.exists(os.path.join(ehd_dir, date)):
                os.mkdir(os.path.join(ehd_dir, date))
            if not os.path.exists(os.path.join(ehd_dir, date, new_project_name)):
                os.mkdir(os.path.join(ehd_dir, date, new_project_name))

            wav_files = [f.name for f in os.scandir(project_dir) if f.name.endswith(".WAV")]
            hprj_file = f"{project_name}.hprj"
            for fname in wav_files + [hprj_file]:
                fp = os.path.join(project_dir, fname)
                if os.path.exists(fp):
                    print(f"found file: {fp}")
                    assert project_name in fname, fname
                    target_fname = fname.replace(project_name, new_project_name)
                    target_fp = os.path.join(ehd_dir, date, new_project_name, target_fname)
                    assert not os.path.exists(target_fp), f"file exists: {target_fp}"
                    shutil.move(fp, target_fp)  # os.rename doesn't work because it wants to make a cross-device link
                    print(f"moved {fp} --> {target_fp}")
                else:
                    print(f"WARNING: no such file: {fp}")
