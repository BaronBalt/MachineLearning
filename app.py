import os
from api.server import app

if __name__ == "__main__":

    debug = os.getenv("APP_ENV", "development") != "production"

    app.run(debug=debug, host="0.0.0.0", port=5000)
