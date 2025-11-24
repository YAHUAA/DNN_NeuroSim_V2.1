#include <vector>
// 将上一级文件夹加入搜索路径

#ifndef ConctrateUnit_H_
#define ConctrateUnit_H_

#include "../typedef.h"
#include "../InputParameter.h"
#include "../Technology.h"
#include "../MemCell.h"
#include "../FunctionUnit.h"

#include "../DFF.h"

using namespace std;

class ContrateUnit: public FunctionUnit {
public:
    ContrateUnit(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell);
    //virtual ~ContrateUnit() {}
    const InputParameter& inputParameter;
    const Technology& tech;
    const MemCell& cell;

    /* Functions */
    void PrintProperty(const char* str);
    void Initialize(int _numUnit, int _numBit, double _clkFreq);
    void CalculateArea(double _newHeight, double _newWidth, AreaModify _option);
    void CalculateLatency(double numShiftBits);
    void CalculatePower(double numShiftBits);
    void CalculateUnitArea();

    /* Properties */
    bool initialized;	/* Initialization flag */
    int numUnit;
    int numDff;
    int numBit;
    double clkFreq;

    DFF dff;
};


#endif /* BITSHIFTER_H_ */