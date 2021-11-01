package spawnn.utils;

import java.util.List;

public class DataFrame {
	public enum binding {
		Integer, Double, Long
	};

	public List<double[]> samples;		// numeric samples
	public List<String> names; 
	public List<binding> bindings;
	
	public String[] getNames() {
		return names.toArray(new String[]{});
	}
	
	public int size() {
		return samples.size();
	}
}
