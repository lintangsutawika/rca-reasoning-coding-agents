import re
import subprocess

def create_localtunnel(port=8080):
    """
    Create a localtunnel for the specified port and return the URL.
    
    Args:
        port (int): The local port to expose (default: 8080)
    
    Returns:
        str: The localtunnel URL
    
    Raises:
        RuntimeError: If localtunnel fails to start or URL cannot be extracted
    """
    try:
        # Start localtunnel process
        process = subprocess.Popen(
            ['npx', 'localtunnel', '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read output line by line to find the URL
        for line in process.stdout:
            # Look for the URL pattern
            match = re.search(r'https://[^\s]+\.loca\.lt', line)
            if match:
                url = match.group(0)
                return url, process
        
        # If we get here, no URL was found
        stderr = process.stderr.read()
        raise RuntimeError(f"Failed to get localtunnel URL. Error: {stderr}")
        
    except FileNotFoundError:
        raise RuntimeError("npx or localtunnel not found. Make sure Node.js is installed.")
    except Exception as e:
        raise RuntimeError(f"Error creating localtunnel: {str(e)}")