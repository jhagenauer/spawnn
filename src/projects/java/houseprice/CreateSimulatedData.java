package houseprice;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.utils.DataUtils;

public class CreateSimulatedData {

	private static Logger log = Logger.getLogger(CreateSimulatedData.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
				
		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("x1");
		vars.add("lnp");
		
		List<double[]> ns = new ArrayList<double[]>();
		while( ns.size() < 1000 ) {
			double xco = r.nextDouble();
			double yco = r.nextDouble();
			
			double x1 = r.nextDouble();
			
			// double lnp = Math.pow(x1, 2); // best for l(max)
			double lnp = Math.pow(x1, 2)*(xco+yco); // best for l(2-3), llm better than linReg
			ns.add( new double[]{xco,yco,x1,lnp});
		}
		
		
		DataUtils.writeCSV("output/houseprice.csv", ns, vars.toArray(new String[]{}));	
	}
}
