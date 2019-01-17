def mean(numbers):
    return sum(numbers)/len(numbers)

def std_dev(numbers):
    avg = mean(numbers)
    variance = sum([(numb - avg) ** 2 for numb in numbers])/len(numbers)
    return variance
