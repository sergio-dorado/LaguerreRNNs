
Command Log
========

## Pendulum (Dynamic System Prediction)

### Pendulum (linear; noiseless)

```
python dyn_pred.py BRC Pendulum 0.00 5 0.8 && python dyn_pred.py nBRC Pendulum 0.00 5 0.8
```

### Pendulum (linear; noisy)

```
python dyn_pred.py BRC Pendulum 0.03 5 0.8 && python dyn_pred.py nBRC Pendulum 0.03 5 0.8
```

### Pendulum (nonlinear; noiseless)

```
python dyn_pred.py BRC Pendulum 0.00 5 2.4 && python dyn_pred.py nBRC Pendulum 0.00 5 2.4
```

### Pendulum (nonlinear; noisy)

```
python dyn_pred.py BRC Pendulum 0.03 5 2.4 && python dyn_pred.py nBRC Pendulum 0.03 5 2.4
```

### All Pendulum Experiments

```
python dyn_pred.py BRC Pendulum 0.00 5 0.8 && python dyn_pred.py nBRC Pendulum 0.00 5 0.8 && python dyn_pred.py BRC Pendulum 0.03 5 0.8 && python dyn_pred.py nBRC Pendulum 0.03 5 0.8 && python dyn_pred.py BRC Pendulum 0.00 5 2.4 && python dyn_pred.py nBRC Pendulum 0.00 5 2.4 && python dyn_pred.py BRC Pendulum 0.03 5 2.4 && python dyn_pred.py nBRC Pendulum 0.03 5 2.4
```

## FluidFlow (Dynamic System Prediction)

### FluidFlow (noiseless)

```
python dyn_pred.py BRC FluidFlow 0.00 5 && python dyn_pred.py nBRC FluidFlow 0.00 5
```

### FluidFlow (noisy)

```
python dyn_pred.py BRC FluidFlow 0.03 5 && python dyn_pred.py nBRC FluidFlow 0.03 5
```

### All FluidFlow Experiments

```
python dyn_pred.py BRC FluidFlow 0.00 5 && python dyn_pred.py nBRC FluidFlow 0.00 5 && python dyn_pred.py BRC FluidFlow 0.03 5 && python dyn_pred.py nBRC FluidFlow 0.03 5
```

## Pendulum (System Identification)

```
python sysid.py BRC Pendulum k-fold verb && python sysid.py nBRC Pendulum k-fold verb
```

## FluidFlow (System Identification)

```
python sysid.py BRC FluidFlow k-fold verb && python sysid.py nBRC FluidFlow k-fold verb
```

## All System Identification Experiments

```
python sysid.py BRC Pendulum k-fold verb && python sysid.py nBRC Pendulum k-fold verb && python sysid.py BRC FluidFlow k-fold verb && python sysid.py nBRC FluidFlow k-fold verb
```

## psMNIST

#### k-fold cross-validation

```
python class.py BRC psMNIST k-fold verb && python class.py nBRC psMNIST k-fold verb
```

#### Normal Splitting

```
python class.py BRC psMNIST normal verb && python class.py nBRC psMNIST normal verb
```

## UCR Datasets

### PenDigits

#### k-fold cross-validation

```
python class.py BRC PenDigits k-fold verb && python class.py nBRC PenDigits k-fold verb
```

#### Normal Splitting

```
python class.py BRC PenDigits normal verb && python class.py nBRC PenDigits normal verb
```

### ChlorineConcentration

#### k-fold cross-validation

```
python class.py BRC ChlorineConcentration k-fold verb && python class.py nBRC ChlorineConcentration k-fold verb
```

#### Normal Splitting

```
python class.py BRC ChlorineConcentration normal verb && python class.py nBRC ChlorineConcentration normal verb
```

### PhonemeSpectra

#### k-fold cross-validation

```
python class.py BRC PhonemeSpectra k-fold verb && python class.py nBRC PhonemeSpectra k-fold verb
```

#### Normal Splitting

```
python class.py BRC PhonemeSpectra normal verb && python class.py nBRC PhonemeSpectra normal verb
```

### Wafer

#### k-fold cross-validation

```
python class.py BRC Wafer k-fold verb && python class.py nBRC Wafer k-fold verb
```

#### Normal Splitting

```
python class.py BRC Wafer normal verb && python class.py nBRC Wafer normal verb
```

### All UCR Experiments (k-fold cross-validation)

```
python class.py BRC PenDigits k-fold verb && python class.py nBRC PenDigits k-fold verb && python class.py BRC ChlorineConcentration k-fold verb && python class.py nBRC ChlorineConcentration k-fold verb && python class.py BRC PhonemeSpectra k-fold verb && python class.py nBRC PhonemeSpectra k-fold verb && python class.py BRC Wafer k-fold verb && python class.py nBRC Wafer k-fold verb
```

### All UCR Experiments (normal splitting)

```
python class.py BRC PenDigits normal verb && python class.py nBRC PenDigits normal verb && python class.py BRC ChlorineConcentration normal verb && python class.py nBRC ChlorineConcentration normal verb && python class.py BRC PhonemeSpectra normal verb && python class.py nBRC PhonemeSpectra normal verb && python class.py BRC Wafer normal verb && python class.py nBRC Wafer normal verb
```
