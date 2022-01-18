
Command Log
========

## Pendulum (Dynamic System Prediction)

### Pendulum (linear; noiseless)

```
python dyn_pred.py Ladder Pendulum 0.00 5 0.8 && python dyn_pred.py Laguerre Pendulum 0.00 5 0.8 && python dyn_pred.py LMU Pendulum 0.00 5 0.8
```

### Pendulum (linear; noisy)

```
python dyn_pred.py Ladder Pendulum 0.03 5 0.8 && python dyn_pred.py Laguerre Pendulum 0.03 5 0.8 && python dyn_pred.py LMU Pendulum 0.03 5 0.8
```

### Pendulum (nonlinear; noiseless)

```
python dyn_pred.py Ladder Pendulum 0.00 5 2.4 && python dyn_pred.py Laguerre Pendulum 0.00 5 2.4 && python dyn_pred.py LMU Pendulum 0.00 5 2.4
```

### Pendulum (nonlinear; noisy)

```
python dyn_pred.py Ladder Pendulum 0.03 5 2.4 && python dyn_pred.py Laguerre Pendulum 0.03 5 2.4 && python dyn_pred.py LMU Pendulum 0.03 5 2.4
```

### All Pendulum Experiments

```
python dyn_pred.py Ladder Pendulum 0.00 5 0.8 && python dyn_pred.py Laguerre Pendulum 0.00 5 0.8 && python dyn_pred.py LMU Pendulum 0.00 5 0.8 && python dyn_pred.py Ladder Pendulum 0.03 5 0.8 && python dyn_pred.py Laguerre Pendulum 0.03 5 0.8 && python dyn_pred.py LMU Pendulum 0.03 5 0.8 && python dyn_pred.py Ladder Pendulum 0.00 5 2.4 && python dyn_pred.py Laguerre Pendulum 0.00 5 2.4 && python dyn_pred.py LMU Pendulum 0.00 5 2.4 && python dyn_pred.py Ladder Pendulum 0.03 5 2.4 && python dyn_pred.py Laguerre Pendulum 0.03 5 2.4 && python dyn_pred.py LMU Pendulum 0.03 5 2.4
```

## FluidFlow (Dynamic System Prediction)

### FluidFlow (noiseless)

```
python dyn_pred.py Ladder FluidFlow 0.00 5 && python dyn_pred.py Laguerre FluidFlow 0.00 5 && python dyn_pred.py LMU FluidFlow 0.00 5
```

### FluidFlow (noisy)

```
python dyn_pred.py Ladder FluidFlow 0.03 5 && python dyn_pred.py Laguerre FluidFlow 0.03 5 && python dyn_pred.py LMU FluidFlow 0.03 5
```

### All FluidFlow Experiments

```
python dyn_pred.py Ladder FluidFlow 0.00 5 && python dyn_pred.py Laguerre FluidFlow 0.00 5 && python dyn_pred.py LMU FluidFlow 0.00 5 && python dyn_pred.py Ladder FluidFlow 0.03 5 && python dyn_pred.py Laguerre FluidFlow 0.03 5 && python dyn_pred.py LMU FluidFlow 0.03 5
```

## Pendulum (System Identification)

```
python sysid.py Ladder Pendulum k-fold verb && python sysid.py Laguerre Pendulum k-fold verb && python sysid.py LMU Pendulum k-fold verb
```

## FluidFlow (System Identification)

```
python sysid.py Ladder FluidFlow k-fold verb && python sysid.py Laguerre FluidFlow k-fold verb && python sysid.py LMU FluidFlow k-fold verb
```

## All System Identification Experiments

```
python sysid.py Ladder Pendulum k-fold verb && python sysid.py Laguerre Pendulum k-fold verb && python sysid.py LMU Pendulum k-fold verb && python sysid.py Ladder FluidFlow k-fold verb && python sysid.py Laguerre FluidFlow k-fold verb && python sysid.py LMU FluidFlow k-fold verb
```

## psMNIST

#### k-fold cross-validation

```
python class.py Ladder psMNIST k-fold verb && python class.py Laguerre psMNIST k-fold verb && python class.py LMU psMNIST k-fold verb
```

#### Normal Splitting

```
python class.py Ladder psMNIST normal verb && python class.py Laguerre psMNIST normal verb && python class.py LMU psMNIST normal verb
```

## UCR Datasets

### PenDigits

#### k-fold cross-validation

```
python class.py Ladder PenDigits k-fold verb && python class.py Laguerre PenDigits k-fold verb && python class.py LMU PenDigits k-fold verb
```

#### Normal Splitting

```
python class.py Ladder PenDigits normal verb && python class.py Laguerre PenDigits normal verb && python class.py LMU PenDigits normal verb
```

### ChlorineConcentration

#### k-fold cross-validation

```
python class.py Ladder ChlorineConcentration k-fold verb && python class.py Laguerre ChlorineConcentration k-fold verb && python class.py LMU ChlorineConcentration k-fold verb
```

#### Normal Splitting

```
python class.py Ladder ChlorineConcentration normal verb && python class.py Laguerre ChlorineConcentration normal verb && python class.py LMU ChlorineConcentration normal verb
```

### PhonemeSpectra

#### k-fold cross-validation

```
python class.py Ladder PhonemeSpectra k-fold verb && python class.py Laguerre PhonemeSpectra k-fold verb && python class.py LMU PhonemeSpectra k-fold verb
```

#### Normal Splitting

```
python class.py Ladder PhonemeSpectra normal verb && python class.py Laguerre PhonemeSpectra normal verb && python class.py LMU PhonemeSpectra normal verb
```

### Wafer

#### k-fold cross-validation

```
python class.py Ladder Wafer k-fold verb && python class.py Laguerre Wafer k-fold verb && python class.py LMU Wafer k-fold verb
```

#### Normal Splitting

```
python class.py Ladder Wafer normal verb && python class.py Laguerre Wafer normal verb && python class.py LMU Wafer normal verb
```

### All UCR Experiments (k-fold cross-validation)

```
python class.py Ladder PenDigits k-fold verb && python class.py Laguerre PenDigits k-fold verb && python class.py LMU PenDigits k-fold verb && python class.py Ladder ChlorineConcentration k-fold verb && python class.py Laguerre ChlorineConcentration k-fold verb && python class.py LMU ChlorineConcentration k-fold verb && python class.py Ladder PhonemeSpectra k-fold verb && python class.py Laguerre PhonemeSpectra k-fold verb && python class.py LMU PhonemeSpectra k-fold verb && python class.py Ladder Wafer k-fold verb && python class.py Laguerre Wafer k-fold verb && python class.py LMU Wafer k-fold verb
```

### All UCR Experiments (normal splitting)

```
python class.py Ladder PenDigits normal verb && python class.py Laguerre PenDigits normal verb && python class.py LMU PenDigits normal verb && python class.py Ladder ChlorineConcentration normal verb && python class.py Laguerre ChlorineConcentration normal verb && python class.py LMU ChlorineConcentration normal verb && python class.py Ladder PhonemeSpectra normal verb && python class.py Laguerre PhonemeSpectra normal verb && python class.py LMU PhonemeSpectra normal verb && python class.py Ladder Wafer normal verb && python class.py Laguerre Wafer normal verb && python class.py LMU Wafer normal verb
```
