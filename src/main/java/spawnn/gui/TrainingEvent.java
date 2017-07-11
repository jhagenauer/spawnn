package spawnn.gui;

import java.util.EventObject;
import java.util.List;

public class TrainingEvent extends EventObject {

	private static final long serialVersionUID = 4058741401226662412L;
	private List<double[]> samples;
	private boolean wmc = false;

	public TrainingEvent(Object source, List<double[]> samples, boolean wmc ) {
		super(source);
		this.samples = samples;
		this.wmc = wmc;
	}
	
	public List<double[]> getSamples() {
		return samples;
	}
	
	public boolean isWMC() {
		return wmc;
	}
}
