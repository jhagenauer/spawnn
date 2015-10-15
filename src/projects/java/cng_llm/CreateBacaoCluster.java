package cng_llm;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class CreateBacaoCluster {

	public static void main(String[] args) {
		try {
			Random r = new Random();
			FileWriter fw = new FileWriter("output/bacao.csv");
			fw.write("lat,lon,x1,x2,y,c\n");
			for (int i = 0; i < 1000; i++) {
				double lat = r.nextDouble();
				double lon = r.nextDouble();

				double y;
				double x1 = r.nextDouble();
				double x2 = r.nextDouble();
				int c;

				// since all cluster have same mean, clustering y is not sufficient. Coefficients must be considered
				if (lon < 0.3) {
					if (r.nextDouble() < 0.0)
						y = x1;
					else
						y = x2;
					c = 0;
				} else if (lon > 0.6) {
					if (r.nextDouble() < 0.0)
						y = x1;
					else
						y = x2;
					c = 1;
					y+=1;
				} else {  // mittel
					if (r.nextDouble() < 1.0)
						y = x1;
					else
						y = x2;
					c = 2;
					y+=2;
				}
				fw.write(lat + "," + lon + "," + x1 + "," + x2 + "," + y + ","+c+"\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
