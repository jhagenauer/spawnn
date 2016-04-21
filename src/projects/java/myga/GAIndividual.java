package myga;

public abstract class GAIndividual<T> implements Comparable<T>{
	private double value = 0;
	
	public abstract void mutate();
	public abstract T recombine( T mother );
	
	public void setValue(double value) {
		this.value = value;
	}
	
	public double getValue() {
		return value;
	}
	
	@Override
	public int compareTo(T o) {
		return Double.compare(value, ((GAIndividual<T>) o).getValue());
	}
}
