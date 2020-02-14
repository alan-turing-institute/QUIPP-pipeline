import re
import subprocess


def generate_provenance_json(script="unknown", params={}):
    """Generate the provenance in a format which can later be output as valid json.

    Inputs:
        string: The name of the script used to trigger the data generation/deidentification/synthesis process
        dict: The parameters used to tune the data generation etc. process; should include random seeds and other
              options as appropriate for the method

    Returns:
        dict: Details of the script called by the user and any relevant parameters
    """

    commit = get_git_commit_hash()
    local_mods = get_local_changes()

    provenance = {"script": script,
                  "commit": commit,
                  "local_modifications": local_mods,
                  "parameters": params}

    return provenance


def get_git_commit_hash():
    """Get the hash of the latest commit in the directory from which this command was called.

    Returns:
        string: The hash of the latest commit
    """

    # Use git rev-parse to try to get the current hash, then use regex to check its format
    # The 7-length has that we ask for should be fine, and if it goes over then well check for that in the regex
    # (git rev-parse will apparently return as many characters as needed for a unique short hash)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"]).strip().decode()
    except subprocess.CalledProcessError:
        return "unknown"

    match = re.fullmatch(r"[a-z0-9]{7,10}", revision)

    # Default to "unknown" if the string returned by the git command isn't in the expected format
    if match is None:
        return "unknown"
    else:
        return revision


def get_local_changes():
    """Determine whether local changes have been made to the current git repository

    Returns:
        bool or None: bool indicating presence of modifications or None if not run in a git repository
        bool or None: bool indicating presence of untracked files or None if not run in a git repository
    """

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).strip().decode()
    except subprocess.CalledProcessError:
        return None, None

    local_mods = False

    # If there have been changes, each line will start with a one or two character key indicating type of change
    if status == "":
        local_mods = False
    else:
        for line in status.splitlines():
            if re.match("[MADRCU]{1,5}", line):
                local_mods = True
            elif re.match("\?\?", line):    # Indicates untracked files - we don't need to record this
                continue
            else:
                print("Unexpected start of line - does this indicate a local modification?")
                local_mods = True           # play it safe and indicate that there are local changes

    return local_mods


if __name__ == "__main__":

    p = generate_provenance_json()
    print(p)
