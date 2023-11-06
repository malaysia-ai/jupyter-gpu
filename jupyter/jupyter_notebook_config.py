c.ServerProxy.servers = {
    'vscode': {
        'command': ['code-server', '--auth=none', '--bind-addr=localhost:{port}'],
        'timeout': 60,
        'launcher_entry': {
            'title': 'VS Code'
        },
    },
}
