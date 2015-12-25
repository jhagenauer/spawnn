package spawnn.utils;

import java.util.List;

public class DataFrame {
	public enum binding {
		Integer, Double, Long
	};

	public List<double[]> samples;
	public List<String> names; // is String[] better?!
	public List<binding> bindings;
	
	public String[] getNames() {
		return names.toArray(new String[]{});
	}
}
