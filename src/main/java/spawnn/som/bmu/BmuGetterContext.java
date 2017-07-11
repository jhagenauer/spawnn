package spawnn.som.bmu;


// Works for somsd as well as hsom
public abstract class BmuGetterContext extends BmuGetter<double[]> {
	public abstract double[] getContext(double[] d); 
}
