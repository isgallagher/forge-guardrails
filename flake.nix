{
  description = "Forge Guardrails dev shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
      in {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
            (python.withPackages (ps: with ps; [
              pip
              pytest
              pytest-asyncio
              pytest-cov
              ruff
            ]))
          ];

          shellHook = ''
            # Create venv if it does not exist
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi
            source .venv/bin/activate
            pip install -q -e ".[dev]"
          '';
        };
      }
    );
}
