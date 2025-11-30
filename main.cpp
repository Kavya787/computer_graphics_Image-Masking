#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/colon.h>
#include <igl/slice.h>
#include <igl/upsample.h>
#include <igl/decimate.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/harmonic.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Tri;

void printHelpExit() {
	printf("USAGE: Inpaint [options]\n\n");

	printf("OPTIONS: \n");

	printf("  -in\t\t\tMesh File with a hole\n");
	printf("  -out\t\t\toutput mesh file\n");
	printf("  -outfaces\t\tFaces in output\n");
	printf("  -upsample\t\tUpsampling\n");

	exit(1);
}

const char* FindCommandLineParam(const char* param, int argc, char* argv[]) {
	const char* token = nullptr;
	for (int i = 0; i < argc; ++i) {
		if (strcmp(argv[i], param) == 0) {
			if (i + 1 < argc) {
				token = argv[i + 1];
				break;
			}
		}
	}

	if (token == nullptr) {
		printf("Could not find command-line parameter %s\n", param);
		return nullptr;
	}

	return token;
}

const char* parseStringParam(const char* param, int argc, char* argv[]) {
	const char* token = FindCommandLineParam(param, argc, argv);
	return token;
}

bool parseIntParam(const char* param, int argc, char* argv[], unsigned int& out) {
	const char* token = FindCommandLineParam(param, argc, argv);
	if (token == nullptr)
		return false;

	int r = sscanf(token, "%u,", &out);
	if (r != 1 || r == EOF) {
		return false;
	}
	else {
		return true;
	}
}

int main(int argc, char *argv[])
{
	
	unsigned int outFacesN;
	unsigned int upsampleN;

	const char* inFile = parseStringParam("-in", argc, argv);
	if (inFile == nullptr){
		printf("1\n");
		printHelpExit();
	}
	const char* outFile = parseStringParam("-out", argc, argv);
	if (outFile == nullptr){
		printf("2\n");
		printHelpExit();
	}
	if (!parseIntParam("-outfaces", argc, argv, outFacesN)){
		printf("3\n");
		printHelpExit();
	}
	if (!parseIntParam("-upsample", argc, argv, upsampleN)){
		printf("4\n");
		printHelpExit();
	}
	//Declaration of Storing Variables

	MatrixXd vertices;
	MatrixXi faces;
    MatrixXd Texture_Coordinates;
    MatrixXi Faces_Texture_Coordinates; 
    MatrixXd Normals;
    MatrixXi Normal_Faces;

	//Reading the Mess

	if (!igl::readOBJ(inFile, vertices, Texture_Coordinates, Normals, faces, Faces_Texture_Coordinates, Normal_Faces)) {
		printf("5\n");
		printHelpExit();
	}

	//Package to Read the 
	
	VectorXi originalLoop; 
	igl::boundary_loop(faces, originalLoop);

	if (originalLoop.size() == 0) {
		printf("Mesh has no hole!");
		printHelpExit();
	}

	igl::upsample(Eigen::MatrixXd(vertices), Eigen::MatrixXi(faces), vertices, faces, upsampleN);


	VectorXd bcenter(3);
	{
		VectorXi R = originalLoop;
		VectorXi C(3); C << 0, 1, 2;
		MatrixXd B;
		MatrixXd A = vertices;
		igl::slice(A, R, C, B);
		bcenter = (1.0f / originalLoop.size()) * B.colwise().sum();
	}

	MatrixXd patchV = MatrixXd(originalLoop.size() + 1, 3); 
	MatrixXi patchF = MatrixXi(originalLoop.size(), 3);

	{
		VectorXi R = originalLoop;
		VectorXi C(3); C << 0, 1, 2;
		MatrixXd A = vertices;
		MatrixXd temp1;
		igl::slice(A, R, C, temp1);

		MatrixXd temp2(1, 3);
		temp2 << bcenter(0), bcenter(1), bcenter(2);

		igl::cat(1, temp1, temp2, patchV);

		for (int i = 0; i < originalLoop.size(); ++i) {
			patchF(i, 2) = (int)originalLoop.size();
			patchF(i, 1) = i;
			patchF(i, 0) = (1 + i) % originalLoop.size();
		}


		igl::upsample(Eigen::MatrixXd(patchV), Eigen::MatrixXi(patchF), patchV, patchF, upsampleN);
	}

	std::vector<std::vector<double>> fusedV;
	std::vector<std::vector<int>> fusedF;

	int index = 0; 

	{
		for (; index < patchV.rows(); ++index) {
			fusedV.push_back({ patchV(index, 0), patchV(index, 1), patchV(index, 2) });
		}

		int findex = 0;
		for (; findex < patchF.rows(); ++findex) {
			fusedF.push_back({ patchF(findex, 0), patchF(findex, 1), patchF(findex, 2) });
		}
	}

	{
		
		std::map<int, int> originalToFusedMap;

		for (int itri = 0; itri < faces.rows(); ++itri) {

			int triIndices[3];
			for (int iv = 0; iv < 3; ++iv) {

				int triIndex = faces(itri, iv);

				int ret;

				if (originalToFusedMap.count(triIndex) == 0) {
					int foundMatch = -1;
					for (int jj = 0; jj < patchV.rows(); ++jj) {
						VectorXd u(3); u << fusedV[jj][0], fusedV[jj][1], fusedV[jj][2];
						VectorXd v(3); v << vertices(triIndex, 0), vertices(triIndex, 1), vertices(triIndex, 2);

						if ((u - v).norm() < 0.00001) {
							foundMatch = jj;
							break;
						}
					}

					if (foundMatch != -1) {
						originalToFusedMap[triIndex] = foundMatch;
						ret = foundMatch;
					}
					else {
						fusedV.push_back({ vertices(triIndex, 0), vertices(triIndex, 1), vertices(triIndex, 2) });
						originalToFusedMap[triIndex] = index;
						ret = index;
						index++;
					}
				}
				else {
					ret = originalToFusedMap[triIndex];
				}

				triIndices[iv] = ret;
			}

			fusedF.push_back({
				triIndices[0],
				triIndices[1],
				triIndices[2] });

		}

	}

	MatrixXd fairedV(fusedV.size(), 3);
	MatrixXi fairedF(fusedF.size(), 3);
	
	{

		for (int vindex = 0; vindex < fusedV.size(); ++vindex) {
			auto r = fusedV[vindex];

			fairedV(vindex, 0) = r[0];
			fairedV(vindex, 1) = r[1];
			fairedV(vindex, 2) = r[2];
		}

		for (int findex = 0; findex < fusedF.size(); ++findex) {
			auto r = fusedF[findex];

			fairedF(findex, 0) = r[0];
			fairedF(findex, 1) = r[1];
			fairedF(findex, 2) = r[2];
		}

		VectorXi b(fairedV.rows() - patchV.rows());
		MatrixXd bc(fairedV.rows() - patchV.rows(), 3);
		for (int i = (int)patchV.rows(); i < (int)fairedV.rows(); ++i) {
			int jj = i - (int)patchV.rows();

			b(jj) = i;

			bc(jj, 0) = fairedV(i, 0);
			bc(jj, 1) = fairedV(i, 1);
			bc(jj, 2) = fairedV(i, 2);
		}

		MatrixXd Z;
		int k = 2;
		igl::harmonic(fairedV, fairedF, b, bc, k, Z);
		fairedV = Z;
	}

	MatrixXd finalV(fusedV.size(), 3);
	MatrixXi finalF(fusedF.size(), 3);
	VectorXi temp0; VectorXi temp1;
	igl::decimate(fairedV, fairedF, outFacesN, finalV, finalF, temp0, temp1);
	igl::writeOBJ(outFile, finalV, finalF);
}