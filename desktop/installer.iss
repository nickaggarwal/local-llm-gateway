; Inno Setup script for the Windows installer.
; Compiled by desktop/build_windows.ps1 after the PyInstaller build.

[Setup]
AppName=Local LLM Gateway
AppVersion=1.0.0
AppPublisher=Local LLM Gateway
DefaultDirName={autopf}\Local LLM Gateway
DefaultGroupName=Local LLM Gateway
OutputDir=..\dist
OutputBaseFilename=LocalLLMGateway-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayName=Local LLM Gateway

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; Flags: unchecked

[Files]
Source: "..\dist\LocalLLMGateway\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Local LLM Gateway"; Filename: "{app}\LocalLLMGateway.exe"
Name: "{autodesktop}\Local LLM Gateway"; Filename: "{app}\LocalLLMGateway.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\LocalLLMGateway.exe"; Description: "Launch Local LLM Gateway"; Flags: nowait postinstall skipifsilent
