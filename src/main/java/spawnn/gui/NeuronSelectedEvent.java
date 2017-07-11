package spawnn.gui;

import java.util.EventObject;

public class NeuronSelectedEvent <T> extends EventObject {
	private static final long serialVersionUID = 1859409290686106279L;
	
	private T p;

	public T getNeuron() {
		return p;
	}

	public NeuronSelectedEvent(Object source, T p ) {
		super(source);
		this.p = p;
	}
}
