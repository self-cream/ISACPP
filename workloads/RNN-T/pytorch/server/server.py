from flask import Flask, jsonify, request
import os

shared_data = {}

def read_shared_data_file():
    with open("/workspace/rnnt/shared_data.txt", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            shared_data[key] = value


app = Flask(__name__)

# Define an API endpoint to retrieve environment variable values
@app.route('/get_env_variable')
def get_env_variable():
    variable_name = request.args.get('name')
    read_shared_data_file()
    variable_value = shared_data[variable_name]
    return jsonify({variable_name: variable_value})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
