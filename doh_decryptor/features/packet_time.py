from datetime import datetime
import numpy
from scipy import stats as stat


class PacketTime:
    """This class extracts features related to the Packet Times."""
    count = 0

    def __init__(self, flow):
        self.flow = flow
        PacketTime.count += 1
        self.packet_times = None

    def _get_packet_times(self):
        """Gets a list of the times of the packets on a flow

        Returns:
            A list of the packet times.

        """
        if self.packet_times is not None:
            return self.packet_times

        # PyShark packets use sniff_timestamp which is a string, need to convert to float
        first_packet_time = float(self.flow.packets[0][0].sniff_timestamp)
        packet_times = [float(packet.sniff_timestamp) - first_packet_time for packet, _ in self.flow.packets]
        return packet_times

    def relative_time_list(self):
        relative_time_list = []
        packet_times = self._get_packet_times()
        for index, time in enumerate(packet_times):
            if index == 0:
                relative_time_list.append(0)
            elif index < len(packet_times):
                relative_time_list.append(float(time - packet_times[index - 1]))
            elif index < 50:
                relative_time_list.append(0)
            else:
                break

        return relative_time_list

    def get_time_stamp(self):
        """Returns the date and time in a human readable format.

        Return (str):
            String of Date and time.

        """
        time = float(self.flow.packets[0][0].sniff_timestamp)
        date_time = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
        return date_time

    def get_duration(self):
        """Calculates the duration of a network flow.

        Returns:
            The duration of a network flow.

        """
        packet_times = self._get_packet_times()
        if not packet_times:
            return 0
        return max(packet_times) - min(packet_times)

    def get_var(self):
        """Calculates the variation of packet times in a network flow.

        Returns:
            float: The variation of packet times.

        """
        packet_times = self._get_packet_times()
        if not packet_times:
            return 0
        return numpy.var(packet_times)

    def get_std(self):
        """Calculates the standard deviation of packet times in a network flow.

        Returns:
            float: The standard deviation of packet times.

        """
        return numpy.sqrt(self.get_var())

    def get_mean(self):
        """Calculates the mean of packet times in a network flow.

        Returns:
            float: The mean of packet times

        """
        packet_times = self._get_packet_times()
        if not packet_times:
            return 0
        return numpy.mean(packet_times)

    def get_median(self):
        """Calculates the median of packet times in a network flow.

        Returns:
            float: The median of packet times

        """
        packet_times = self._get_packet_times()
        if not packet_times:
            return 0
        return numpy.median(packet_times)

    def get_mode(self):
        """The mode of packet times in a network flow.

        Returns:
            float: The mode of packet times

        """
        packet_times = self._get_packet_times()
        if not packet_times:
            return -1
        return float(stat.mode(packet_times)[0])

    def get_skew(self):
        """Calculates the skew of packet times in a network flow using the median.

        Returns:
            float: The skew of packet times.

        """
        mean = self.get_mean()
        median = self.get_median()
        dif = 3 * (mean - median)
        std = self.get_std()

        if std == 0:
            return -10
        return dif / std

    def get_skew2(self):
        """Calculates the skew of the packet times in a network flow using the mode.

        Returns:
            float: The skew of the packet times.

        """
        mean = self.get_mean()
        mode = self.get_mode()
        dif = (float(mean) - mode)
        std = self.get_std()

        if std == 0:
            return -10
        return dif / float(std)

    def get_cov(self):
        """Calculates the coefficient of variance of packet times in a network flow.

        Returns:
            float: The coefficient of variance of a packet times list.

        """
        mean = self.get_mean()
        if mean == 0:
            return -1
        return self.get_std() / mean