package moomap.myga;

public abstract class DefaultIndividual implements GAIndividual {
	protected double value;
	
	@Override
	public double getValue() {	
		return value;
	}
	
	@Override 
	public void setValue(double value) {
		this.value = value;
	}
	
	@Override
	public int compareTo(GAIndividual o) {
		if( getValue() < o.getValue() )
			return -1;
		else if( getValue() > o.getValue() )
			return 1;
		else
			return 0;
	}
	
}
