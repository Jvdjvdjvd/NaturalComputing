with (import <nixpkgs> {});
let extensions = (with pkgs.vscode-extensions; [
      ms-vsliveshare.vsliveshare
      ms-python.python
    ]);
  vscode-with-extensions = pkgs.vscode-with-extensions.override {
      vscodeExtensions = extensions;
    };
in pkgs.mkShell {
  buildInputs = [
    python38
    python38Packages.matplotlib
    python38Packages.numpy
    vscode-with-extensions
  ];
}
