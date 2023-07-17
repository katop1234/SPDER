## Paper

For more information, see our [paper on arXiv](https://arxiv.org/abs/2306.15242).

## Usage

In general, scripts can be run with the following command structure:

```bash
python SPDER_[media].py <file_path> <architecture>
```
or
```bash
python SPDER_[media].py <object_of_the_file> <architecture>
```

### Example

For instance, to run the `SPDER_image_batched.py` script on an object named 'camera' with architecture 'SPDER', use:

```bash
python SPDER_image_batched.py camera SPDER
```

Replace `[media]` in the above commands with the appropriate media type you are using. `<file_path>` and `<object_of_the_file>` should also be replaced with the actual path to your file or the object name in your file respectively. `<architecture>` should be replaced with SPDER/SIREN/RELU/RELU_PE (positional encoding)
