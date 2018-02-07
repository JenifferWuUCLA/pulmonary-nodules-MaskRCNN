"""
Written by Matteo Dunnhofer - 2017

Utilities procedures and Functions
"""

def format_time(time):
	""" It formats a datetime to print it
		Args:
			time: datetime
		Returns:
			a formatted string representing time
	"""
	m, s = divmod(time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))