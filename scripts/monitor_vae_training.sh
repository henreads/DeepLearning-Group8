#!/bin/bash
# Monitor VAE x224 training on Modal

echo "VAE x224 Training Monitor"
echo "=========================="
echo ""
echo "To check training progress:"
echo "  1. Visit: https://modal.com/apps"
echo "  2. Look for app: wafer-defect-vae-x224-main"
echo ""
echo "To download results after training:"
echo "  modal run modal_apps/vae_x224_main/app.py::download_artifacts"
echo ""
echo "Expected training time: 2-3 hours on A10G GPU"
echo ""
echo "Checking Modal volume for artifacts..."
echo ""

modal volume ls wafer-defect-vae-x224-main-artifacts /vae_x224 2>/dev/null || echo "Waiting for artifacts..."
