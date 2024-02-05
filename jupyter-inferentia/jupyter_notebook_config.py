from jupyter_core.paths import jupyter_data_dir
import subprocess
import os
import errno
import stat

c = get_config()
c.NotebookApp.open_browser = False

# https://github.com/jupyter/notebook/issues/3130
c.FileContentsManager.delete_to_trash = False

c.ServerProxy.servers = {
    'code-server': {
        'command': ['code-server', '--auth=none', '--bind-addr=localhost:{port}'],
        'timeout': 60,
        'launcher_entry': {
            'title': 'VS Code'
        },
    },
}