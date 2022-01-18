
Command Log
========

## Pendulum (Dynamic System Prediction)

### Pendulum (linear; noiseless)

```
python dyn_pred.py GRU Pendulum 0.00 5 0.8 && python dyn_pred.py LSTM Pendulum 0.00 5 0.8
```

### Pendulum (linear; noisy)

```
python dyn_pred.py GRU Pendulum 0.03 5 0.8 && python dyn_pred.py LSTM Pendulum 0.03 5 0.8
```

### Pendulum (nonlinear; noiseless)

```
python dyn_pred.py GRU Pendulum 0.00 5 2.4 && python dyn_pred.py LSTM Pendulum 0.00 5 2.4
```

### Pendulum (nonlinear; noisy)

```
python dyn_pred.py GRU Pendulum 0.03 5 2.4 && python dyn_pred.py LSTM Pendulum 0.03 5 2.4
```

### All Pendulum Experiments

```
python dyn_pred.py GRU Pendulum 0.00 5 0.8 && python dyn_pred.py LSTM Pendulum 0.00 5 0.8 && python dyn_pred.py GRU Pendulum 0.03 5 0.8 && python dyn_pred.py LSTM Pendulum 0.03 5 0.8 && python dyn_pred.py GRU Pendulum 0.00 5 2.4 && python dyn_pred.py LSTM Pendulum 0.00 5 2.4 && python dyn_pred.py GRU Pendulum 0.03 5 2.4 && python dyn_pred.py LSTM Pendulum 0.03 5 2.4
```

## FluidFlow (Dynamic System Prediction)

### FluidFlow (noiseless)

```
python dyn_pred.py GRU FluidFlow 0.00 5 && python dyn_pred.py LSTM FluidFlow 0.00 5
```

### FluidFlow (noisy)

```
python dyn_pred.py GRU FluidFlow 0.03 5 && python dyn_pred.py LSTM FluidFlow 0.03 5
```

### All FluidFlow Experiments

```
python dyn_pred.py GRU FluidFlow 0.00 5 && python dyn_pred.py LSTM FluidFlow 0.00 5 && python dyn_pred.py GRU FluidFlow 0.03 5 && python dyn_pred.py LSTM FluidFlow 0.03 5
```

## Pendulum (System Identification)

```
python sysid.py GRU Pendulum k-fold verb && python sysid.py LSTM Pendulum k-fold verb
```

## FluidFlow (System Identification)

```
python sysid.py GRU FluidFlow k-fold verb && python sysid.py LSTM FluidFlow k-fold verb
```

## All System Identification Experiments

```
python sysid.py GRU Pendulum k-fold verb && python sysid.py LSTM Pendulum k-fold verb && python sysid.py GRU FluidFlow k-fold verb && python sysid.py LSTM FluidFlow k-fold verb
```

## psMNIST

#### k-fold cross-validation

```
python class.py GRU psMNIST k-fold verb && python class.py LSTM psMNIST k-fold verb
```

#### Normal Splitting

```
python class.py GRU psMNIST normal verb && python class.py LSTM psMNIST normal verb
```

## UCR Datasets

### PenDigits

#### k-fold cross-validation

```
python class.py GRU PenDigits k-fold verb && python class.py LSTM PenDigits k-fold verb
```

#### Normal Splitting

```
python class.py GRU PenDigits normal verb && python class.py LSTM PenDigits normal verb
```

### ChlorineConcentration

#### k-fold cross-validation

```
python class.py GRU ChlorineConcentration k-fold verb && python class.py LSTM ChlorineConcentration k-fold verb
```

#### Normal Splitting

```
python class.py GRU ChlorineConcentration normal verb && python class.py LSTM ChlorineConcentration normal verb
```

### PhonemeSpectra

#### k-fold cross-validation

```
python class.py GRU PhonemeSpectra k-fold verb && python class.py LSTM PhonemeSpectra k-fold verb
```

#### Normal Splitting

```
python class.py GRU PhonemeSpectra normal verb && python class.py LSTM PhonemeSpectra normal verb
```

### Wafer

#### k-fold cross-validation

```
python class.py GRU Wafer k-fold verb && python class.py LSTM Wafer k-fold verb
```

#### Normal Splitting

```
python class.py GRU Wafer normal verb && python class.py LSTM Wafer normal verb
```

### All UCR Experiments (k-fold cross-validation)

```
python class.py GRU PenDigits k-fold verb && python class.py LSTM PenDigits k-fold verb && python class.py GRU ChlorineConcentration k-fold verb && python class.py LSTM ChlorineConcentration k-fold verb && python class.py GRU PhonemeSpectra k-fold verb && python class.py LSTM PhonemeSpectra k-fold verb && python class.py GRU Wafer k-fold verb && python class.py LSTM Wafer k-fold verb
```

### All UCR Experiments (normal splitting)

```
python class.py GRU PenDigits normal verb && python class.py LSTM PenDigits normal verb && python class.py GRU ChlorineConcentration normal verb && python class.py LSTM ChlorineConcentration normal verb && python class.py GRU PhonemeSpectra normal verb && python class.py LSTM PhonemeSpectra normal verb && python class.py GRU Wafer normal verb && python class.py LSTM Wafer normal verb
```
