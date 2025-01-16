import numpy
from scipy import stats as stat
from doh_decryptor.context.packet_direction import PacketDirection


class ResponseTime:
    """A summary of features based on the time difference
       between an outgoing packet and the following response.
    """

    def __init__(self, feature):
        self.feature = feature

    def get_dif(self) -> list:
        """Calculates the time difference in seconds between
           an outgoing packet and the following response packet.

        Returns:
            List[float]: A list of time differences.
        """
        time_diff = []
        temp_packet = None
        temp_direction = None

        for packet, direction in self.feature.packets:
            if temp_direction == PacketDirection.FORWARD and direction == PacketDirection.REVERSE:
                # Convert pyshark timestamps to float before calculation
                current_time = float(packet.sniff_timestamp)
                temp_time = float(temp_packet.sniff_timestamp)
                time_diff.append(current_time - temp_time)
            temp_packet = packet
            temp_direction = direction

        return time_diff

    def get_var(self) -> float:
        """Calculates the variation of the list of time differences.

        Returns:
            float: The variation in time differences.
        """
        diffs = self.get_dif()
        if not diffs:
            return -1
        return numpy.var(diffs)

    def get_mean(self) -> float:
        """Calculates the mean of the list of time differences.

        Returns:
            float: The mean in time differences.
        """
        diffs = self.get_dif()
        if not diffs:
            return -1
        return numpy.mean(diffs)

    def get_median(self) -> float:
        """Calculates the median of the list of time differences

        Returns:
            float: The median in time differences.
        """
        diffs = self.get_dif()
        if not diffs:
            return -1
        return numpy.median(diffs)

    def get_mode(self) -> float:
        """Calculates the mode of the of time differences

        Returns:
            float: The mode in time differences.
        """
        diffs = self.get_dif()
        if not diffs:
            return -1
        return float(stat.mode(diffs)[0])

    def get_std(self) -> float:
        """Calculates the standard deviation of the list of time differences

        Returns:
            float: The standard deviation in time differences.
        """
        diffs = self.get_dif()
        if not diffs:
            return -1
        return numpy.sqrt(self.get_var())

    def get_skew(self) -> float:
        """Calculates the skew of the of time differences.

        Note:
            Uses a simple skew formula using the mean and the median.
        Returns:
            float: The skew in time differences.
        """
        mean = self.get_mean()
        median = self.get_median()
        dif = 3 * (mean - median)
        std = self.get_std()

        if std == 0 or mean == -1 or median == -1:
            return -10
        return dif / std

    def get_skew2(self) -> float:
        """Calculates the skew of the of time differences.

        Note:
            Uses a simple skew formula using the mean and the mode
        Returns:
            float: The skew in time differences.
        """
        mean = self.get_mean()
        mode = self.get_mode()
        dif = (float(mean) - mode)
        std = self.get_std()

        if std == 0 or mean == -1 or mode == -1:
            return -10
        return dif / float(std)

    def get_cov(self) -> float:
        """Calculates the coefficient of variance of the list of time differences

        Note:
            return -1 if division by 0.
        Returns:
            float: The coefficient of variance in time differences.
        """
        mean = self.get_mean()
        std = self.get_std()

        if mean == 0 or mean == -1 or std == -1:
            return -1
        return std / mean