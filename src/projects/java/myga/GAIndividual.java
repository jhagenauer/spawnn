package myga;

public abstract class GAIndividual implements Comparable<GAIndividual>{
	private double value = 0;
	
	public abstract void mutate();
	public abstract GAIndividual recombine( GAIndividual mother );
	
	public void setValue(double value) {
		this.value = value;
	}
	
	public double getValue() {
		return value;
	}
	
	@Override
	public int compareTo(GAIndividual o) {
		return Double.compare(value, o.getValue());
	}
}
