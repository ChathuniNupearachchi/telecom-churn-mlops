# Create all directories
$folders = @(
    "data\raw",
    "data\processed",
    "data\validation",
    "notebooks",
    "src\data",
    "src\models",
    "src\monitoring",
    "src\api",
    "tests",
    "config",
    "models",
    ".github\workflows",
    "docker"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Write-Host "✅ Created: $folder"
}

# Create __init__.py files
$initFiles = @(
    "src\__init__.py",
    "src\data\__init__.py",
    "src\models\__init__.py",
    "src\monitoring\__init__.py",
    "src\api\__init__.py",
    "tests\__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path $file | Out-Null
    Write-Host "✅ Created: $file"
}

Write-Host "`n🎉 All folders and files created successfully!"