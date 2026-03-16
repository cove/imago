@echo off
cd /d "%~dp0.."
echo Streamable-HTTP: http://192.168.4.26:8090/mcp
echo Job console:     http://192.168.4.26:8091
echo.
.venv\Scripts\python.exe mcp_server.py --transport http --host 0.0.0.0 --port 8090 --console-host 192.168.4.26
