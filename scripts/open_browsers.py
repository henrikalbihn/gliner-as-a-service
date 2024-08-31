"""
Open the various services in the system's default browser.
"""

import webbrowser

HOSTNAME = "http://localhost"

HOSTS = {
    "fastapi": f"{HOSTNAME}:8080/docs",
    "locust": f"{HOSTNAME}:8089",
    "flower": f"{HOSTNAME}:5556",
}


def main() -> None:
    """Main function"""
    for service, url in HOSTS.items():
        webbrowser.open(url, new=1 if service == "fastapi" else 2)


if __name__ == "__main__":
    main()
