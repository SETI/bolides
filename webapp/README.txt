Maintenance instructions, to run when git repository is updated:
================================================================
ssh into the server
cd bolides
git pull
systemctl restart flask.service


Installation instructions for a public webapp instance:
=======================================================

1) Dependencies
sudo apt install pip

Install Miniconda however you like, then:
conda create -n bolides
conda activate bolides

Follow installation instructions at bolides.readthedocs.io.

pip install plotly dash flask_caching gunicorn
sudo apt install nginx

Download spacekit.js from https://typpo.github.io/spacekit/build/spacekit.js
and place it in bolides/webapp/assets as _spacekit.js

Download sprites from https://github.com/typpo/spacekit/tree/master/src/assets/sprites
and place them in bolides/webapp/assets/sprites

2) Setting up server configs
rm /etc/nginx/sites-enabled/default

Copy the flask.conf in bolides/webapp/configs (also below) to:
/etc/nginx/conf.d/flask.conf
--------------------------------------------------------------------
server {
        server_name bolides.seti.org;
        location / {
                include proxy_params;
                proxy_pass http://127.0.0.1:8050;
        }

        listen 80 default_server;
        listen [::]:80 default_server;
}
--------------------------------------------------------------------

Copy the flask.service in bolides/webapp/configs (also below) to:
/etc/systemd/system/flask.service
This config assumes that the user running the server is "webapp".
-------------------------------------------------------------
[Unit]
Description=Gunicorn instance to serve Flask
After=network.target
[Service]
User=root
Group=www-data
WorkingDirectory=/home/webapp/bolides/webapp
Environment="PATH=/home/webapp/miniconda3/envs/bolides/bin"
ExecStart=/home/webapp/miniconda3/envs/bolides/bin/gunicorn --bind 127.0.0.1:8050 wsgi:app
[Install]
WantedBy=multi-user.target
-------------------------------------------------------------

3) Starting the server

Start the services:
systemctl start flask.service
systemctl start nginx.service
Make them start on restart:
systemctl enable flask.service
systemctl enable nginx.service
