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
    openjdk
    python38
    python38Packages.deap
    python38Packages.pygraphviz
    python38Packages.matplotlib
    python38Packages.numpy
    python38Packages.scipy
    python38Packages.scikitlearn
    vscode-with-extensions
  ];
}
