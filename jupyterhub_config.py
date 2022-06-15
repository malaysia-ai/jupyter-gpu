c.JupyterHub.debug_proxy = True
c.Authenticator.admin_users = {'husein-comel', 'huseinzol05'}
c.JupyterHub.log_level = 'DEBUG'
c.LocalProcessSpawner.debug = True
c.Spawner.debug = True
c.JupyterHub.port = 8010
c.JupyterHub.hub_bind_url = 'http://127.0.0.1:8085'
c.ConfigurableHTTPProxy.api_url = 'http://localhost:8005'
