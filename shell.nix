with ((import (fetchTarball {
  name = "nixpkgs-master-2021-04-09";
  url = "https://github.com/nixos/nixpkgs/archive/cb29de02c4c0e0bcb95ddbd7cc653dd720689bab.tar.gz";
  sha256 = "1daxszcvj3bq6qkki7rfzkd0f026n08xvvfx7gkr129nbcnpg24p";
}) {}));
let
  extensions = (with pkgs.vscode-extensions; [
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
