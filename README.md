# Fluid Dynamics Super-Resolution Neural Network

## Overview

This project implements advanced machine learning techniques for high-fidelity fluid dynamics simulation reconstruction, specifically focusing on the Taylor-Green Vortex (TGV) problem. The repository provides a powerful neural network approach to enhance and reconstruct fluid flow simulations using two state-of-the-art architectures: Super-Resolution Convolutional Neural Network (SRCNN) and Super-Resolution Generative Adversarial Network (SRGAN).

## Features

- 🌊 Fluid Dynamics Simulation Reconstruction
- 🧠 Two Neural Network Architectures:
  - SRCNN (Super-Resolution Convolutional Neural Network)
  - SRGAN (Super-Resolution Generative Adversarial Network)
- 🔬 Supports High-Order (HO) and Low-Order (LO) Simulation Reconstruction
- 📊 Comprehensive Error Analysis
- 🖥️ GPU Acceleration Support

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- External `horses3d.ns` simulation executable

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fluid-dynamics-superres.git
   cd fluid-dynamics-superres
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Configure your neural network parameters in `NEURALNET/config_nn.dat`. Key configuration options include:

- `architecture`: Choose between 'SRCNN' or 'SRGAN'
- `simulation`: Enable/disable external simulation
- `trained_model`: Use existing model or train a new one
- Simulation and training parameters (epochs, batch size, etc.)

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1. Optionally run fluid dynamics simulations
2. Load or train a neural network model
3. Reconstruct simulation data
4. Save results in the `RESULTS/` directory

## Output

The script generates:
- Reconstructed flow variables (rhou, rhov, rhow)
- L2 Error metrics
- Normalized value ranges

## Project Structure

```
├── NEURALNET/
│   ├── src/
│   │   ├── SuperRes_utils.py
│   │   ├── horses3d.py
│   │   ├── cnn.py
│   │   └── gan.py
│   ├── config_nn.dat
│   └── nns/
│       ├── MyModel_SRCNN/
│       └── MyModel_SRGAN/
├── RESULTS/
│   └── solution files
└── Main.py
```

## Supported Flow Variables

- `rhou`: X-direction momentum
- `rhov`: Y-direction momentum
- `rhow`: Z-direction momentum

## Performance

- GPU-accelerated training and inference
- Supports memory-efficient model training
- Fallback to CPU if no GPU is detected

## Limitations

- Requires external `horses3d.ns` simulation executable
- Performance depends on input data quality and simulation parameters

## Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss potential improvements.

## License

[Add your project's license here]

## Citation

If you use this work in academic research, please cite:
[Add citation information if applicable]

## Acknowledgments

[Add any acknowledgments or credits]

   


