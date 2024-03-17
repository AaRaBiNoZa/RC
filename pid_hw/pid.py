# TODO: implement a class for PID controller
class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: add aditional variables to store the current state of the controller
        self.integral_ = 0

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):
        curr_read, last_read = sensor_readings
        last_err = commanded_variable - last_read
        curr_err = commanded_variable - curr_read

        dedt = (curr_err - last_err) / self.sensor_period
        integral = curr_err * self.sensor_period

        self.integral_ += integral

        return self.gain_prop * curr_err + self.gain_int * self.integral_ + self.gain_der * dedt