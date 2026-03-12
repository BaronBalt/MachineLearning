import os

from api import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.getenv("APP_ENV", "development") != "production"
    app.run(debug=debug, host="0.0.0.0", port=5000)
