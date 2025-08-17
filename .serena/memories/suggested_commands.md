# Suggested Commands

## Build Commands

### Build Solution
```batch
"C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\msbuild.exe" needaimbot.sln /p:Configuration=Release /p:Platform=x64 /t:Build
```

### Quick Build (using build.bat)
```batch
build.bat
```

### Clean Build
```batch
"C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\msbuild.exe" needaimbot.sln /p:Configuration=Release /p:Platform=x64 /t:Clean
```

## Development Commands

### Run Application
```batch
cd x64\Release
ai.exe
```

### Debug Mode (if available)
```batch
cd x64\Debug
ai.exe
```

## Version Control Commands

### Check Status
```batch
git status
```

### Create Feature Branch
```batch
git checkout -b feature/your-feature-name
```

### Commit Changes
```batch
git add .
git commit -m "feat: description of changes"
```

### Conventional Commit Examples
```batch
git commit -m "feat: add new tracking algorithm"
git commit -m "fix: resolve memory leak in detector"
git commit -m "perf: optimize color conversion kernel"
git commit -m "docs: update build instructions"
git commit -m "refactor: separate CaptureState from AppContext"
```

## Testing Commands

### Manual Testing
- Run the application and verify basic functionality
- Test different capture methods
- Verify overlay interface works
- Check different input device drivers

### Performance Testing
- Monitor GPU usage with GPU-Z
- Check memory usage with Task Manager
- Verify inference times in overlay

## System Commands (Windows)

### File Operations
```batch
dir                    # List directory contents
cd <directory>         # Change directory
copy <src> <dest>      # Copy files
move <src> <dest>      # Move files
del <file>             # Delete file
```

### Process Management
```batch
tasklist              # List running processes
taskkill /f /im ai.exe # Force kill application
```

### System Information
```batch
systeminfo            # System information
nvidia-smi            # GPU information
```