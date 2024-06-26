#include <cstdlib>

#include <utility>

#include "Definition.h"
#include "IR.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Var.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;

struct DefinitionContents {
    mutable RefCount ref_count;
    bool is_init = true;
    Expr predicate;
    std::vector<Expr> values, args;
    StageSchedule stage_schedule;
    std::vector<Specialization> specializations;

    DefinitionContents()
        : predicate(const_true()) {
    }

    void accept(IRVisitor *visitor) const {
        if (predicate.defined()) {
            predicate.accept(visitor);
        }

        for (const Expr &val : values) {
            val.accept(visitor);
        }
        for (const Expr &arg : args) {
            arg.accept(visitor);
        }

        stage_schedule.accept(visitor);

        for (const Specialization &s : specializations) {
            if (s.condition.defined()) {
                s.condition.accept(visitor);
            }
            s.definition.accept(visitor);
        }
    }

    void mutate(IRMutator *mutator) {
        if (predicate.defined()) {
            predicate = mutator->mutate(predicate);
        }

        for (auto &value : values) {
            value = mutator->mutate(value);
        }
        for (auto &arg : args) {
            arg = mutator->mutate(arg);
        }

        stage_schedule.mutate(mutator);

        for (Specialization &s : specializations) {
            if (s.condition.defined()) {
                s.condition = mutator->mutate(s.condition);
            }
            s.definition.mutate(mutator);
        }
    }
};

template<>
RefCount &ref_count<DefinitionContents>(const DefinitionContents *d) noexcept {
    return d->ref_count;
}

template<>
void destroy<DefinitionContents>(const DefinitionContents *d) {
    delete d;
}

Definition::Definition()
    : contents(nullptr) {
}

Definition::Definition(const IntrusivePtr<DefinitionContents> &ptr)
    : contents(ptr) {
    internal_assert(ptr.defined())
        << "Can't construct Function from undefined DefinitionContents ptr\n";
}

Definition::Definition(const std::vector<Expr> &args, const std::vector<Expr> &values,
                       const ReductionDomain &rdom, bool is_init)
    : contents(new DefinitionContents) {
    contents->is_init = is_init;
    contents->values = values;
    contents->args = args;
    if (rdom.defined()) {
        contents->predicate = rdom.predicate();
        for (const auto &rv : rdom.domain()) {
            contents->stage_schedule.rvars().push_back(rv);
        }
    }
}

Definition::Definition(bool is_init, const Expr &predicate, const std::vector<Expr> &args, const std::vector<Expr> &values,
                       const StageSchedule &schedule, const std::vector<Specialization> &specializations)
    : contents(new DefinitionContents) {
    contents->is_init = is_init;
    contents->values = values;
    contents->args = args;
    contents->predicate = predicate;
    contents->stage_schedule = schedule;
    contents->specializations = specializations;
}

Definition Definition::get_copy() const {
    internal_assert(contents.defined()) << "Cannot copy undefined Definition\n";

    Definition copy(new DefinitionContents);
    copy.contents->is_init = contents->is_init;
    copy.contents->predicate = contents->predicate;
    copy.contents->values = contents->values;
    copy.contents->args = contents->args;
    copy.contents->stage_schedule = contents->stage_schedule.get_copy();

    // Deep-copy specializations
    for (const Specialization &s : contents->specializations) {
        Specialization s_copy;
        s_copy.condition = s.condition;
        s_copy.definition = s.definition.get_copy();
        s_copy.failure_message = s.failure_message;
        copy.contents->specializations.push_back(std::move(s_copy));
    }
    return copy;
}

bool Definition::defined() const {
    return contents.defined();
}

bool Definition::is_init() const {
    return contents->is_init;
}

void Definition::accept(IRVisitor *visitor) const {
    contents->accept(visitor);
}

void Definition::mutate(IRMutator *mutator) {
    contents->mutate(mutator);
}

std::vector<Expr> &Definition::args() {
    return contents->args;
}

const std::vector<Expr> &Definition::args() const {
    return contents->args;
}

std::vector<Expr> &Definition::values() {
    return contents->values;
}

const std::vector<Expr> &Definition::values() const {
    return contents->values;
}

Expr &Definition::predicate() {
    return contents->predicate;
}

const Expr &Definition::predicate() const {
    return contents->predicate;
}

std::vector<Expr> Definition::split_predicate() const {
    std::vector<Expr> predicates;
    split_into_ands(contents->predicate, predicates);
    return predicates;
}

StageSchedule &Definition::schedule() {
    return contents->stage_schedule;
}

const StageSchedule &Definition::schedule() const {
    return contents->stage_schedule;
}

std::vector<Specialization> &Definition::specializations() {
    return contents->specializations;
}

const std::vector<Specialization> &Definition::specializations() const {
    return contents->specializations;
}

const Specialization &Definition::add_specialization(Expr condition) {
    Specialization s;
    s.condition = std::move(condition);
    s.definition.contents = new DefinitionContents;
    s.definition.contents->is_init = contents->is_init;
    s.definition.contents->predicate = contents->predicate;
    s.definition.contents->values = contents->values;
    s.definition.contents->args = contents->args;

    // The sub-schedule inherits everything about its parent except for its specializations.
    s.definition.contents->stage_schedule = contents->stage_schedule.get_copy();

    contents->specializations.push_back(s);
    return contents->specializations.back();
}

}  // namespace Internal
}  // namespace Halide
