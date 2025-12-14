
````
# Hands-On Lab: Docker Fundamentals

This lab is designed to be copy-pasted into your terminal. You will build a
simple Python web application, containerize it, and connect it to a database
(Redis) to understand networking.

## Lab Objective
By the end of this lab, you will have mastered:
1.  **Building** images (`docker build`)
2.  **Running** containers (`docker run`, `docker ps`)
3.  **Debugging** (`docker logs`, `docker exec`)
4.  **Networking** containers together (`docker network`)
5.  **Cleaning up** (`docker stop`, `docker rm`, `docker prune`)

---

## Part 1: Setup the Project

First, create a folder and the necessary files for our "Hit Counter" app.

### 1. Create a directory
```bash
mkdir docker-lab
cd docker-lab
````

### 2\. Create `app.py` (The Application)

Create a file named `app.py` with the following code. This simple Flask app connects to Redis to count page views.

```python
from flask import Flask
import redis
import os

app = Flask(__name__)

# Connect to Redis using the hostname 'my-redis' 
# (We will define this hostname in Docker later)
cache = redis.Redis(host='my-redis', port=6379)

@app.route('/')
def hello():
    try:
        count = cache.incr('hits')
        return f'Hello Docker! I have been seen {count} times.\n'
    except redis.exceptions.ConnectionError:
        return 'Hello Docker! (Redis is not connected yet)\n'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### 3\. Create `requirements.txt` (The Dependencies)

Create a file named `requirements.txt`:

```text
flask
redis
```

### 4\. Create the `Dockerfile` (The Blueprint)

Create a file named `Dockerfile` (no extension):

```dockerfile
# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file first (for caching efficiency)
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code
COPY . .

# Step 6: Define the command to run the app
CMD ["python", "app.py"]
```

-----

## Part 2: Build and Run (Single Container)

### 1\. Build the Image

We will tag (`-t`) the image as `hit-counter:v1`. The `.` tells Docker to look for the Dockerfile in the current directory.

```bash
docker build -t hit-counter:v1 .
```

### 2\. List your Images

Verify the image was created.

```bash
docker images
```

### 3\. Run the Container

We map port **5000** inside the container to port **8080** on your machine.

```bash
docker run -d -p 8080:5000 --name my-web-app hit-counter:v1
```

  * `-d`: Detached mode (runs in background).
  * `--name`: Gives the container a friendly name.

### 4\. Check container status

```bash
docker ps
```

**Action:** Open your browser to `http://<>:8080` <> is your docker enviorinment URL.
**Result:** You should see "Hello Docker\! (Redis is not connected yet)".

### 5\. View Logs

If something goes wrong (or to see the print statements), check the logs.

```bash
docker logs my-web-app
```

-----

## Part 3: Networking (Connecting to a Database)

Right now, the app fails to connect to Redis because Redis isn't running. We need to run Redis and put both containers on the same **Network**.

### 1\. Create a Network

```bash
docker network create lab-net
```

### 2\. Run Redis on the Network

We verify the Redis image is downloaded and run it attached to our new network.

```bash
docker run -d --network lab-net --name my-redis redis:alpine
```

  * **Note:** The container name `my-redis` becomes its DNS name inside this network.

### 3\. Connect the Web App to the Network

Our running web app (`my-web-app`) is not on this network. We could restart it, but let's just connect it live.

```bash
docker network connect lab-net my-web-app
```

### 4\. Verify Connectivity

Refresh `http://localhost:8080`.

  * **Result:** "Hello Docker\! I have been seen 1 times."
  * Refresh again -\> "Seen 2 times".

-----

## Part 4: Debugging & Executing Commands

Sometimes you need to go *inside* the container to check files or run manual commands.

### 1\. Open a Shell inside the container

```bash
docker exec -it my-web-app sh
```

  * You are now inside the Linux environment of the container\!
  * Type `ls` to see your `app.py`.
  * Type `exit` to leave.

### 2\. Inspect the Container

To see technical details (IP address, Environment variables, Mounts).

```bash
docker inspect my-web-app
```

-----

## Part 5: Development Workflow (Bind Mounts)

Currently, if you edit `app.py` on your laptop, the running container doesn't change. You'd have to rebuild. Let's fix that using a **Volume (Bind Mount)**.

### 1\. Stop and Remove the old container

```bash
docker stop my-web-app
docker rm my-web-app
```

### 2\. Run with a Bind Mount

This maps your current folder (`$(pwd)`) to `/app` inside the container.
*(Note: On Windows PowerShell, replace `$(pwd)` with `${PWD}`)*.

```bash
docker run -d \
  -p 8080:5000 \
  --network lab-net \
  --name my-web-app \
  -v $(pwd):/app \
  hit-counter:v1
```

### 3\. Test Live Updates

1.  Open `app.py` in your text editor.
2.  Change "Hello Docker\!" to "Hello **MASTER**\!".
3.  Save the file.
4.  Restart the container to pick up the change (Flask requires restart unless in debug mode):
    ```bash
    docker restart my-web-app
    ```
5.  Refresh `http://<>:8080`. <> is your Docker enviorinment URL You should see the text change **without rebuilding the image**.

-----

## Part 6: Cleanup

Never leave unused containers eating up your RAM\!

### 1\. Stop and Remove containers

```bash
docker stop my-web-app my-redis
docker rm my-web-app my-redis
```

### 2\. Remove the Network

```bash
docker network rm lab-net
```

### 3\. Nuclear Option (Optional)

This command deletes all stopped containers, unused networks, and dangling images.

```bash
docker system prune -f
```

-----

## Summary of Commands Used

| Command | Description |
| :--- | :--- |
| `docker build -t <name> .` | Build an image from a Dockerfile. |
| `docker images` | List available images. |
| `docker run -d -p <host>:<container> <img >` | Run a container in background with port mapping. |
| `docker ps` | List running containers. |
| `docker logs <name>` | View stdout/stderr logs of a container. |
| `docker network create <net>` | Create a virtual network for container communication. |
| `docker exec -it <name> sh` | Open an interactive shell inside a container. |
| `docker stop <name>` | Gracefully stop a container. |
| `docker rm <name>` | Delete a stopped container. |
| `docker system prune` | Clean up unused resources. |

```
```

