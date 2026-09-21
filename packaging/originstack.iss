; Inno Setup script for the OriginStack Windows installer.
;
; Wraps the PyInstaller onedir build (packaging\dist\OriginStack\) in a normal Setup.exe:
; one double-click copies the whole app (OriginStack.exe *and* its _internal\ folder --
; running the exe from inside the zip preview, which extracts only the exe, fails with
; "Failed to load Python DLL"), adds a Start Menu entry and an uninstaller.
;
; Build with packaging\build_installer.ps1, which passes the version from the VERSION file:
;   iscc /DAppVersion=2.1.0 packaging\originstack.iss
; Installs per user (no admin prompt) into %LOCALAPPDATA%\Programs\OriginStack; the wizard
; offers an all-users install through the usual "install for me only / all users" dialog.
; Inno Setup is a build-time tool only -- nothing here adds a runtime dependency.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{4D3DE3BB-8C04-4520-A51B-EAAE8DCC0FD8}
AppName=OriginStack
AppVersion={#AppVersion}
AppVerName=OriginStack {#AppVersion}
AppPublisher=OriginStack
AppPublisherURL=https://github.com/hd152/originstack
AppSupportURL=https://github.com/hd152/originstack/issues
AppUpdatesURL=https://github.com/hd152/originstack/releases
DefaultDirName={autopf}\OriginStack
DefaultGroupName=OriginStack
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=OriginStack-{#AppVersion}-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\OriginStack.exe
UninstallDisplayName=OriginStack
LicenseFile=..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; a running copy would leave locked files behind
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\OriginStack\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\OriginStack"; Filename: "{app}\OriginStack.exe"
Name: "{autodesktop}\OriginStack"; Filename: "{app}\OriginStack.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OriginStack.exe"; Description: "Launch OriginStack"; Flags: nowait postinstall skipifsilent
