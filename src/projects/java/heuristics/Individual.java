package heuristics;

public abstract class Individual<T> implements Comparable<T>{
	private double value = 0;
	
	public void setValue(double value) {
		this.value = value;
	}
	
	public double getValue() {
		return value;
	}
	
	@Override
	public int compareTo(T o) {
		return Double.compare(value, ((Individual<T>) o).getValue());
	}
}
