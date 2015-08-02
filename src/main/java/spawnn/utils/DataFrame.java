package spawnn.utils;

import java.util.List;

public class DataFrame {
	public enum binding {
		Integer, Double, Long
	};

	public List<double[]> samples;
	public List<String> names;
	public List<binding> bindings;
}
